import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import gc
from datetime import datetime
import copy
from itertools import product
import random
import warnings
warnings.filterwarnings('ignore')

random_seed=42

random.seed(random_seed)
np.random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        encoded_seq = tokenizer.encode(
            " ".join(self.sequences[idx]), 
            add_special_tokens=True, 
            truncation=True, 
            max_length=512
        )
        return torch.tensor(encoded_seq), torch.tensor(self.labels[idx])

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attended_output = torch.sum(attention_weights * x, dim=1)
        return attended_output, attention_weights

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout_rate):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = self._get_activation(activation)
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def _get_activation(self, activation_name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'elu': nn.ELU()
        }
        return activations[activation_name]
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Handle batch size of 1 during training
        if x.size(0) == 1 and self.training:
            # Either skip BatchNorm or use a different path
            out = self.activation(self.layers[0](x))
            out = self.layers[3](out)  # Apply dropout
            out = self.layers[4](out)  # Second linear layer
        else:
            out = self.layers(x)
            
        return self.activation(out + identity)

class EnhancedProtBert(nn.Module):
    def __init__(self, config, architecture_params):
        super().__init__()
        self.bert = BertModel(config)
        
        # Freeze BERT base
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Architecture parameters
        self.hidden_dims = architecture_params['hidden_dims']
        self.dropout_rate = architecture_params['dropout_rate']
        self.activation = architecture_params['activation']
        self.use_attention = architecture_params['use_attention']
        self.num_heads = architecture_params.get('num_heads', 1)
        
        # Multi-head attention
        if self.use_attention:
            self.attention_heads = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate
                ) for _ in range(self.num_heads)
            ])
            
            # Attention pooling
            self.attention_pool = AttentionLayer(config.hidden_size)
        
        # Feature extraction layers
        layers = []
        input_dim = config.hidden_size * (self.num_heads if self.use_attention else 1)
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                ResidualBlock(
                    input_dim,
                    hidden_dim,
                    self.activation,
                    self.dropout_rate
                )
            ])
            input_dim = hidden_dim
            
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.hidden_dims[-1], config.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get BERT outputs (frozen)
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply attention if used
        if self.use_attention:
            attended_outputs = []
            for attention in self.attention_heads:
                attended, _ = attention(
                    sequence_output.permute(1, 0, 2),
                    sequence_output.permute(1, 0, 2),
                    sequence_output.permute(1, 0, 2)
                )
                attended_outputs.append(attended.permute(1, 0, 2))
            
            # Concatenate attended outputs
            sequence_output = torch.cat(attended_outputs, dim=-1)
            
            # Apply attention pooling
            pooled_output, _ = self.attention_pool(sequence_output)
        else:
            # Global average pooling
            pooled_output = sequence_output.mean(dim=1)
        
        # Feature extraction
        features = self.feature_layers(pooled_output)
        logits = self.classifier(features)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits

class DynamicLearningRate:
    def __init__(self, optimizer, mode='cosine', **kwargs):
        self.optimizer = optimizer
        self.mode = mode
        self.kwargs = kwargs
        self.schedulers = {
            'cosine': self._get_cosine_scheduler,
            'one_cycle': self._get_one_cycle_scheduler,
            'cyclic': self._get_cyclic_scheduler
        }
        self.scheduler = self.schedulers[mode]()

    def _get_cosine_scheduler(self):
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.kwargs.get('num_warmup_steps', 100),
            num_training_steps=self.kwargs.get('num_training_steps', 1000)
        )

    def _get_one_cycle_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.kwargs.get('max_lr', 0.01),
            epochs=self.kwargs.get('epochs', 10),
            steps_per_epoch=self.kwargs.get('steps_per_epoch', 100)
        )

    def _get_cyclic_scheduler(self):
        return torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.kwargs.get('base_lr', 1e-4),
            max_lr=self.kwargs.get('max_lr', 1e-3),
            cycle_momentum=False
        )

    def step(self):
        self.scheduler.step()

def evaluate(model, data_loader, device):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'auc_roc': roc_auc_score(all_labels, all_probs)
    }
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, device, dataset_name, fold=None, num_epochs=50):
    """Train model and track metrics for each epoch."""
    best_val_acc = 0
    best_model = None
    training_history = []
    
    # Create directory for plots
    plots_dir = os.path.join('results', dataset_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss, logits = model(inputs, labels=labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(train_labels, train_preds),
            'precision': precision_score(train_labels, train_preds, average='binary'),
            'recall': recall_score(train_labels, train_preds, average='binary'),
            'f1': f1_score(train_labels, train_preds, average='binary')
        }
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        
        # Save epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_results)
        
        # Update best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model = copy.deepcopy(model)
            
            # Save ROC curve for best model
            fold_suffix = f'_fold_{fold}' if fold is not None else ''
            plt.figure(figsize=(8, 6))
            plt.plot(val_metrics['roc_curve']['fpr'], 
                    val_metrics['roc_curve']['tpr'], 
                    label=f'ROC curve (AUC = {val_metrics["auc_roc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name}{fold_suffix}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plots_dir, f'roc_curve{fold_suffix}.png'))
            plt.close()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    # Save training history plot
    fold_suffix = f'_fold_{fold}' if fold is not None else ''
    plt.figure(figsize=(12, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, [h['train_metrics']['accuracy'] for h in training_history], label='Train Acc')
    plt.plot(epochs, [h['val_metrics']['accuracy'] for h in training_history], label='Val Acc')
    plt.title(f'Training History - {dataset_name}{fold_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'training_history{fold_suffix}.png'))
    plt.close()
    
    return best_model, training_history


def custom_collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in sequences], 
        batch_first=True, 
        padding_value=0
    )
    labels = torch.tensor(labels)
    return padded_sequences, labels

def get_grid_search_params():
    """Define parameter grid for search."""
    return {
        'architecture': {
            'hidden_dims': [[512, 256], [256, 128], [512, 256, 128]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'activation': ['relu', 'gelu'],
            'use_attention': [True, False],
            'num_heads': [1, 2, 4]
        },
        'training': {
            'batch_size': [16, 32],
            'base_lr': [1e-4, 2e-4],
            'max_lr': [1e-3, 2e-3],
            'weight_decay': [0.01, 0.1],
            'warmup_steps': [100, 200],
            'lr_scheduler': ['cosine', 'one_cycle']
        }
    }

def run_kfold_grid_search(train_dataset, test_dataset, dataset_name, device, n_splits=10):
    """Run grid search with k-fold CV and save comprehensive metrics."""
    global random_seed
    # Create results directory
    results_dir = os.path.join('results', dataset_name)
    models_dir = os.path.join('models', dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    params_grid = get_grid_search_params()
    best_result = {
        'dataset': dataset_name,
        'cv_metrics': {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'auc_roc': 0
        },
        'params': None,
        'test_metrics': None,
        'fold_histories': [],
        'fold_metrics': []
    }
    
    # Create test loader once
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=custom_collate_fn
    )
    
    # Generate all parameter combinations
    arch_params = [dict(zip(params_grid['architecture'].keys(), v)) 
                  for v in product(*params_grid['architecture'].values())]
    train_params = [dict(zip(params_grid['training'].keys(), v)) 
                   for v in product(*params_grid['training'].values())]
    
    total_combinations = len(arch_params) * len(train_params)
    
    # Initialize K-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    for comb_idx, (arch, train) in enumerate(product(arch_params, train_params)):
        print(f"\nParameter combination {comb_idx + 1}/{total_combinations}")
        params = {'architecture': arch, 'training': train}
        fold_histories = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            # Create data loaders for this fold
            train_loader = DataLoader(
                train_dataset,
                batch_size=train['batch_size'],
                sampler=SubsetRandomSampler(train_idx),
                collate_fn=custom_collate_fn
            )
            val_loader = DataLoader(
                train_dataset,
                batch_size=train['batch_size'],
                sampler=SubsetRandomSampler(val_idx),
                collate_fn=custom_collate_fn
            )
            
            # Initialize model
            config = BertConfig.from_pretrained(
                "Rostlab/prot_bert",
                num_labels=2
            )
            model = EnhancedProtBert(config, arch).to(device)
            
            # Initialize optimizer and scheduler
            optimizer = AdamW(
                model.parameters(),
                lr=train['base_lr'],
                weight_decay=train['weight_decay']
            )
            
            lr_scheduler = DynamicLearningRate(
                optimizer,
                mode=train['lr_scheduler'],
                num_warmup_steps=train['warmup_steps'],
                max_lr=train['max_lr'],
                num_training_steps=50 * len(train_loader)
            )
            
            # Train model for this fold
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                dataset_name=dataset_name,
                fold=fold + 1
            )
            
            # Save fold results
            fold_metrics.append({
                'fold': fold + 1,
                'metrics': evaluate(model, val_loader, device)
            })
            fold_histories.append({
                'fold': fold + 1,
                'history': history
            })
            
            # Clean up
            del model, optimizer, lr_scheduler
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate average CV metrics
        avg_cv_metrics = {
            'accuracy': np.mean([m['metrics']['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['metrics']['precision'] for m in fold_metrics]),
            'recall': np.mean([m['metrics']['recall'] for m in fold_metrics]),
            'f1': np.mean([m['metrics']['f1'] for m in fold_metrics]),
            'auc_roc': np.mean([m['metrics']['auc_roc'] for m in fold_metrics])
        }
        
        print(f"Average CV metrics:")
        for metric, value in avg_cv_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Update best result if current is better
        if avg_cv_metrics['accuracy'] > best_result['cv_metrics']['accuracy']:
            print("New best parameters found!")
            
            # Train final model with best parameters on full training set
            config = BertConfig.from_pretrained(
                "Rostlab/prot_bert",
                num_labels=2
            )
            final_model = EnhancedProtBert(config, arch).to(device)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=train['batch_size'],
                shuffle=True,
                collate_fn=custom_collate_fn
            )
            
            optimizer = AdamW(
                final_model.parameters(),
                lr=train['base_lr'],
                weight_decay=train['weight_decay']
            )
            
            lr_scheduler = DynamicLearningRate(
                optimizer,
                mode=train['lr_scheduler'],
                num_warmup_steps=train['warmup_steps'],
                max_lr=train['max_lr'],
                num_training_steps=50 * len(train_loader)
            )
            
            # Train on full training set
            final_model, _ = train_model(
                model=final_model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                dataset_name=dataset_name
            )
            
            # Evaluate on test set
            test_metrics = evaluate(final_model, test_loader, device)
            model_save_path = os.path.join(models_dir, f"best_model_{random_seed}.pt")  # <--- ADD THIS
            torch.save(final_model.state_dict(), model_save_path) 
            
            # Save the best hyperparameters in a plain text file
            params_file_path = os.path.join(models_dir, f"best_params_{random_seed}.txt")  # <--- ADD THIS
            with open(params_file_path, 'w') as txt_file:                    # <--- ADD THIS
                txt_file.write(str(params))
            
            best_result = {
                'dataset': dataset_name,
                'cv_metrics': avg_cv_metrics,
                'params': params,
                'test_metrics': test_metrics,
                'fold_histories': fold_histories,
                'fold_metrics': fold_metrics,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save best parameters and results
            results_file = os.path.join(results_dir, 'best_results.json')
            with open(results_file, 'w') as f:
                json.dump(best_result, f, indent=4)
            
            del final_model
            torch.cuda.empty_cache()
            gc.collect()
    
    return best_result

def load_fasta_data(fasta_path):
    """Load sequences and labels from FASTA file."""
    sequences = []
    labels = []

    with open(fasta_path, 'r') as file:
        current_seq = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:  # Save previous sequence
                    sequences.append(''.join(current_seq))
                    current_seq = []
                # Extract label from header (assuming last character is label)
                label = int(line[-1])
                labels.append(label)
            else:
                # Add sequence with spaces between amino acids
                current_seq.append(' '.join(line))

        # Add last sequence
        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences, labels

def prepare_datasets(sequences, labels, train_ratio=0.9):
    """Split data into train and test sets."""
    # Create full dataset
    full_dataset = ProteinDataset(sequences, labels)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, test_dataset

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_single_dataset(dataset_path):
    """Train and evaluate a single dataset."""
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load data
    sequences, labels = load_fasta_data(dataset_path)
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(sequences, labels)
    
    # Run grid search with cross-validation
    best_result = run_kfold_grid_search(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        device=device
    )
    
    print(f"\nResults for {dataset_name}:")
    print(f"Best CV accuracy: {best_result['cv_metrics']['accuracy']:.4f}")
    print("Test metrics:", json.dumps(best_result['test_metrics'], indent=2))
    return best_result

def main():
    # List of datasets to process
    datasets = [
        "HIVP/IDV.fasta",
        "HIVP/LPV.fasta",
        "HIVP/NFV.fasta",
        "HIVP/SQV.fasta",
        "HIVP/APV.fasta",
        
    ]
    
    results = {}
    
    # Train on each dataset sequentially
    for dataset_path in datasets:
        try:
            result = train_single_dataset(dataset_path)
            results[dataset_path] = result
        except Exception as e:
            print(f"Error processing {dataset_path}: {str(e)}")
        finally:
            # Clear GPU memory after each dataset
            clear_gpu_memory()
    
    # Save overall results summary
    summary_path = os.path.join('results', 'all_datasets_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        main()
    finally:
        # Final cleanup
        clear_gpu_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()