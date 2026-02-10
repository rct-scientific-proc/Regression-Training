import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SimpleCNN
from loss import CustomLoss
from dataset import ImageRegressionDataset
from training_utils import train_epoch, validate


def create_optimizer(optimizer_name, model_params, trial, lr):
    """Create optimizer with Optuna-suggested hyperparameters.
    
    Args:
        optimizer_name: Name of optimizer (adam, adamw, rmsprop, sgd, adagrad, adamax, nadam)
        model_params: Model parameters to optimize
        trial: Optuna trial object (None if not using Optuna)
        lr: Learning rate
    
    Returns:
        Optimizer instance
    """
    if trial is not None:
        # Optuna is tuning hyperparameters
        if optimizer_name.lower() == 'adam':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = trial.suggest_float('beta2', 0.99, 0.999)
            return optim.Adam(model_params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        
        elif optimizer_name.lower() == 'adamw':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = trial.suggest_float('beta2', 0.99, 0.999)
            return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        
        elif optimizer_name.lower() == 'sgd':
            momentum = trial.suggest_float('momentum', 0.5, 0.99)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            nesterov = trial.suggest_categorical('nesterov', [True, False])
            return optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        
        elif optimizer_name.lower() == 'rmsprop':
            alpha = trial.suggest_float('alpha', 0.9, 0.99)
            momentum = trial.suggest_float('momentum', 0.0, 0.5)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            return optim.RMSprop(model_params, lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay)
        
        elif optimizer_name.lower() == 'adagrad':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            return optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
        
        elif optimizer_name.lower() == 'adamax':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = trial.suggest_float('beta2', 0.99, 0.999)
            return optim.Adamax(model_params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        
        elif optimizer_name.lower() == 'nadam':
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            beta1 = trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = trial.suggest_float('beta2', 0.99, 0.999)
            return optim.NAdam(model_params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    else:
        # Use default parameters when not using Optuna
        if optimizer_name.lower() == 'adam':
            return optim.Adam(model_params, lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(model_params, lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(model_params, lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == 'adagrad':
            return optim.Adagrad(model_params, lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == 'adamax':
            return optim.Adamax(model_params, lr=lr, weight_decay=1e-4)
        elif optimizer_name.lower() == 'nadam':
            return optim.NAdam(model_params, lr=lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, val_patience=None):
    """Train model and return best validation loss."""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update learning rate scheduler (call after optimizer.step() in train_epoch)
        scheduler.step()
        
        # Early stopping
        if val_patience is not None and patience_counter >= val_patience:
            print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break
    
    return best_val_loss, train_losses, val_losses


def objective(trial, train_loader, val_loader, device, epochs, val_patience, channels, input_size, min_feature_size, num_conv_layers, num_fc_layers, optimizer_name):
    """Optuna objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
    # Create model with suggested dropout
    model = SimpleCNN(input_channels=channels, image_size=input_size, dropout_rate=dropout_rate, min_feature_size=min_feature_size, num_conv_layers=num_conv_layers, num_fc_layers=num_fc_layers).to(device)
    
    # Create optimizer with suggested hyperparameters (trial-based)
    optimizer = create_optimizer(optimizer_name, model.parameters(), trial, lr)
    criterion = CustomLoss()
    
    # Train model
    best_val_loss, _, _ = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, val_patience)
    
    return best_val_loss


def main(argv):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model to predict R coefficient from images.")
    parser.add_argument('--csv-file', type=str, required=True, help="Path to the CSV file containing image paths and labels for training.")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to save the training results. Figures, logs, checkpoints, and final best model will be saved here.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (only used if optuna-iterations=0).")
    parser.add_argument('--train-split', type=float, default=0.4, help="Fraction of data to use for training.")
    parser.add_argument('--val-split', type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument('--optuna-iterations', type=int, default=0, help="Number of Optuna hyperparameter optimization iterations. If 0, Optuna is disabled.")
    parser.add_argument('--val-patience', type=int, default=None, help="Number of epochs with no improvement after which training will be stopped (early stopping). If None, early stopping is disabled.")
    parser.add_argument('--channels', type=int, default=1, help="Number of input channels (1 for grayscale, 3 for RGB).")
    parser.add_argument('--input-size', type=int, nargs=2, default=[256, 256], metavar=('WIDTH', 'HEIGHT'), help="Input image size as width and height (default: 256 256).")
    parser.add_argument('--min-feature-size', type=int, default=4, help="Minimum feature map size after final adaptive pooling (default: 4).")
    parser.add_argument('--num-conv-layers', type=int, default=4, help="Number of convolutional layers (default: 4).")
    parser.add_argument('--num-fc-layers', type=int, default=2, help="Number of fully connected hidden layers (default: 2).")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'adamax', 'nadam'], help="Optimizer to use (default: adam).")
    
    args = parser.parse_args(argv)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transform for resizing images if needed
    input_size = min(args.input_size)  # Use minimum dimension for model calculation (assumes roughly square images)
    if args.input_size[0] != 256 or args.input_size[1] != 256:
        transform = transforms.Compose([
            transforms.Resize((args.input_size[1], args.input_size[0]))  # (height, width) for PIL
        ])
    else:
        transform = None
    
    # Load dataset
    print(f"Loading data from {args.csv_file}...")
    full_dataset = ImageRegressionDataset(args.csv_file, num_channels=args.channels, transform=transform)
    
    # Split dataset with stratification based on label values
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Extract all labels for stratification
    all_labels = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    all_labels = np.array(all_labels)
    
    # Create label bins for stratification (using percentile-based bins)
    # This ensures each partition has similar distribution of label values
    n_bins = min(10, total_size // 30)  # Create bins with at least ~30 samples per bin
    label_bins = np.array(pd.cut(all_labels, bins=n_bins, labels=False, duplicates='drop'))
    
    # First split: separate test set
    indices = np.arange(total_size)
    train_val_indices, test_indices, train_val_labels, test_labels = sklearn_train_test_split(
        indices, label_bins,
        test_size=test_size / total_size,
        stratify=label_bins,
        random_state=42
    )
    
    # Second split: separate train and validation from train_val
    train_indices, val_indices, _, _ = sklearn_train_test_split(
        train_val_indices, train_val_labels,
        test_size=val_size / (train_size + val_size),
        stratify=train_val_labels,
        random_state=42
    )
        # Convert indices to numpy arrays for proper indexing
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Convert indices to numpy arrays for proper indexing
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Label distribution - Train: [{all_labels[train_indices].min():.3f}, {all_labels[train_indices].max():.3f}], "
          f"Val: [{all_labels[val_indices].min():.3f}, {all_labels[val_indices].max():.3f}], "
          f"Test: [{all_labels[test_indices].min():.3f}, {all_labels[test_indices].max():.3f}]")
    
    # Create dataloaders (fixed batch size for now)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Best hyperparameters (default or from Optuna)
    best_params = {
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'dropout_rate': 0.3
    }
    
    # Optuna hyperparameter optimization
    if args.optuna_iterations > 0:
        print(f"\n{'='*60}")
        print(f"Starting Optuna hyperparameter optimization ({args.optuna_iterations} iterations)...")
        print(f"{'='*60}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, device, args.epochs, args.val_patience, args.channels, input_size, args.min_feature_size, args.num_conv_layers, args.num_fc_layers, args.optimizer),
            n_trials=args.optuna_iterations,
            show_progress_bar=True
        )
        
        # Get best hyperparameters
        best_params['learning_rate'] = study.best_params['learning_rate']
        best_params['weight_decay'] = study.best_params['weight_decay']
        best_params['dropout_rate'] = study.best_params['dropout_rate']
        
        print(f"\n{'='*60}")
        print(f"Best hyperparameters found:")
        print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
        print(f"  Weight Decay: {best_params['weight_decay']:.6f}")
        print(f"  Dropout Rate: {best_params['dropout_rate']:.4f}")
        print(f"  Best Val Loss: {study.best_value:.6f}")
        print(f"{'='*60}\n")
        
        # Save Optuna study results
        optuna_df = study.trials_dataframe()
        optuna_df.to_csv(os.path.join(args.output_dir, 'optuna_trials.csv'), index=False)
        print(f"Optuna trials saved to {os.path.join(args.output_dir, 'optuna_trials.csv')}")
    
    # Train final model with best hyperparameters
    print(f"\nTraining final model with best hyperparameters...")
    model = SimpleCNN(input_channels=args.channels, image_size=input_size, dropout_rate=best_params['dropout_rate'], min_feature_size=args.min_feature_size, num_conv_layers=args.num_conv_layers, num_fc_layers=args.num_fc_layers).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = CustomLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters(), None, best_params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    print(f"Starting training for {args.epochs} epochs...\n")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update learning rate scheduler (call after optimizer.step() in train_epoch)
        scheduler.step()
        
        # Early stopping
        if args.val_patience is not None and patience_counter >= args.val_patience:
            print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'))
    print(f"\nTraining curve saved to {os.path.join(args.output_dir, 'training_curve.png')}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Save test predictions
    model.eval()
    all_predictions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            
            # Handle case where squeeze removes batch dimension (single sample)
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_images.extend(images.cpu().numpy())
            
            # Labels are already normalized in the dataset
            labels_np = labels.numpy().flatten()
            all_labels.extend(labels_np)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'predicted': all_predictions,
        'actual': all_labels
    })
    predictions_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
    print(f"Test predictions saved to {os.path.join(args.output_dir, 'test_predictions.csv')}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Actual (Normalized)')
    plt.ylabel('Predicted (Normalized)')
    plt.title('Test Set: Predictions vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'predictions_scatter.png'))
    print(f"Predictions scatter plot saved to {os.path.join(args.output_dir, 'predictions_scatter.png')}")
    
    # Plot test set error analysis
    # errors = np.array(all_predictions) - np.array(all_labels)
    # Call the loss function to compute the errors
    errors = []
    for pred, actual in zip(all_predictions, all_labels):
        pred_tensor = torch.tensor(pred).unsqueeze(0).to(device)
        actual_tensor = torch.tensor(actual).unsqueeze(0).to(device)
        error = criterion(pred_tensor, actual_tensor).item()
        errors.append(error)
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error histogram
    axes[0, 0].hist(errors, bins=90, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Error (Predicted - Actual)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Errors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error vs actual label
    axes[0, 1].scatter(all_labels, errors, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Zero error')
    axes[0, 1].set_xlabel('Actual (Normalized)')
    axes[0, 1].set_ylabel('Error (Predicted - Actual)')
    axes[0, 1].set_title('Error vs Actual Label')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Absolute error histogram
    axes[1, 0].hist(abs_errors, bins=90, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Absolute Errors')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error statistics
    axes[1, 1].axis('off')
    stats_text = f"""Test Set Error Statistics
    
Mean Error: {np.mean(errors):.6f}
Std Dev of Error: {np.std(errors):.6f}
Min Error: {np.min(errors):.6f}
Max Error: {np.max(errors):.6f}

Mean Absolute Error: {np.mean(abs_errors):.6f}
Median Absolute Error: {np.median(abs_errors):.6f}
95th Percentile Error: {np.percentile(abs_errors, 95):.6f}

Total Test Samples: {len(errors)}
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'test_error_analysis.png'), dpi=100)
    print(f"Test error analysis saved to {os.path.join(args.output_dir, 'test_error_analysis.png')}")
    
    # Plot worst predictions (highest errors)
    worst_idx = np.argsort(abs_errors)[-9:][::-1]  # Top 9 worst predictions
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(worst_idx):
        ax = axes[i]
        
        # Get image (convert from (C, H, W) to (H, W, C) for display)
        img_array = all_images[idx]
        if img_array.shape[0] == 3:  # RGB image
            img_display = np.transpose(img_array, (1, 2, 0))
            # Normalize to [0, 1] for display if needed
            if img_display.max() > 1.0:
                img_display = img_display / 255.0
        else:
            img_display = img_array[0]  # Grayscale
        
        ax.imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
        
        pred_val = all_predictions[idx]
        actual_val = all_labels[idx]
        error_val = errors[idx]
        abs_error_val = abs_errors[idx]
        
        # Title with predictions
        title = f"Pred: {pred_val:.3f}\nActual: {actual_val:.3f}\nError: {error_val:.3f}\nAbs Error: {abs_error_val:.3f}"
        ax.set_title(title, fontsize=10, color='red' if abs_error_val > 0.1 else 'black')
        ax.axis('off')
    
    plt.suptitle('Top 9 Worst Predictions (Highest Absolute Errors)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'worst_predictions.png'), dpi=100, bbox_inches='tight')
    print(f"Worst predictions figure saved to {os.path.join(args.output_dir, 'worst_predictions.png')}")
    
    # Save best hyperparameters
    with open(os.path.join(args.output_dir, 'best_hyperparameters.txt'), 'w') as f:
        f.write(f"Best Hyperparameters\n")
        f.write(f"====================\n\n")
        f.write(f"Learning Rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"Weight Decay: {best_params['weight_decay']:.6f}\n")
        f.write(f"Dropout Rate: {best_params['dropout_rate']:.4f}\n")
        f.write(f"\nBest Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
    
    # Plot label distribution across train/val/test splits
    # Extract labels from datasets directly
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    train_labels = np.array(train_labels)
    
    val_labels = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        val_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    val_labels = np.array(val_labels)
    
    test_labels = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        test_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    test_labels = np.array(test_labels)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Train histogram
    axes[0].hist(train_labels, bins=90, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Label Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Train Set Distribution (n={len(train_labels)})')
    axes[0].set_xlim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Validation histogram
    axes[1].hist(val_labels, bins=90, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Label Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Validation Set Distribution (n={len(val_labels)})')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # Test histogram
    axes[2].hist(test_labels, bins=90, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Label Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Test Set Distribution (n={len(test_labels)})')
    axes[2].set_xlim([0, 1])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'label_distribution.png'), dpi=100)
    print(f"Label distribution histogram saved to {os.path.join(args.output_dir, 'label_distribution.png')}")
    
    # Create overlay histogram for comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(train_labels, bins=90, alpha=0.5, label=f'Train (n={len(train_labels)})', color='blue', edgecolor='black')
    ax.hist(val_labels, bins=90, alpha=0.5, label=f'Val (n={len(val_labels)})', color='green', edgecolor='black')
    ax.hist(test_labels, bins=90, alpha=0.5, label=f'Test (n={len(test_labels)})', color='red', edgecolor='black')
    ax.set_xlabel('Label Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Label Distribution Comparison (Train vs Val vs Test)')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'label_distribution_overlay.png'), dpi=100)
    print(f"Label distribution overlay histogram saved to {os.path.join(args.output_dir, 'label_distribution_overlay.png')}")
    
    # Save label distribution statistics to JSON
    def compute_label_stats(labels):
        """Compute statistics for a label distribution."""
        return {
            'count': int(len(labels)),
            'mean': float(np.mean(labels)),
            'std': float(np.std(labels)),
            'min': float(np.min(labels)),
            'q25': float(np.percentile(labels, 25)),
            'median': float(np.median(labels)),
            'q75': float(np.percentile(labels, 75)),
            'max': float(np.max(labels)),
        }
    
    label_distribution_stats = {
        'train': compute_label_stats(train_labels),
        'validation': compute_label_stats(val_labels),
        'test': compute_label_stats(test_labels),
        'overall': compute_label_stats(np.concatenate([train_labels, val_labels, test_labels]))
    }
    
    stats_json_path = os.path.join(args.output_dir, 'label_distribution_stats.json')
    with open(stats_json_path, 'w') as f:
        json.dump(label_distribution_stats, f, indent=2)
    print(f"Label distribution statistics saved to {stats_json_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
