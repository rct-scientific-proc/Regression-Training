import argparse
import sys
import os
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model import SimpleCNN
from loss import CustomLoss
from dataset import ImageRegressionDataset
from training_utils import train_epoch, validate
from normalize import normalize_label


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


def objective(trial, train_loader, val_loader, device, epochs, val_patience):
    """Optuna objective function for hyperparameter optimization."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size_mult = trial.suggest_int('batch_size_mult', 1, 4)  # Scale batch size
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
    # Create model with suggested dropout
    model = SimpleCNN(input_channels=3, image_size=256, dropout_rate=dropout_rate).to(device)
    
    # Create optimizer with suggested hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    
    args = parser.parse_args(argv)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from {args.csv_file}...")
    full_dataset = ImageRegressionDataset(args.csv_file)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
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
            lambda trial: objective(trial, train_loader, val_loader, device, args.epochs, args.val_patience),
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
    model = SimpleCNN(input_channels=3, image_size=256, dropout_rate=best_params['dropout_rate']).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
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
            
            # Normalize labels for comparison
            labels_np = labels.numpy().reshape(-1, 1)
            labels_normalized = normalize_label(labels_np).flatten()
            all_labels.extend(labels_normalized)
    
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
    axes[0, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
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
    axes[1, 0].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
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
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
