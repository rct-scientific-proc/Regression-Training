import torch
from normalize import normalize_label


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: Training dataloader
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to run on (cuda/cpu)
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Normalize labels
        labels_np = labels.cpu().numpy().reshape(-1, 1)
        labels_normalized = normalize_label(labels_np)
        labels_normalized = torch.from_numpy(labels_normalized).to(device).float().squeeze()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Ensure shapes match
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        if labels_normalized.dim() == 0:
            labels_normalized = labels_normalized.unsqueeze(0)
        
        # Compute differentiable circular loss
        loss = criterion(outputs, labels_normalized)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / max(batch_count, 1)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on (cuda/cpu)
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Normalize labels
            labels_np = labels.cpu().numpy().reshape(-1, 1)
            labels_normalized = normalize_label(labels_np)
            labels_normalized = torch.from_numpy(labels_normalized).to(device).float().squeeze()
            
            # Forward pass
            outputs = model(images).squeeze()
            
            # Ensure shapes match
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels_normalized.dim() == 0:
                labels_normalized = labels_normalized.unsqueeze(0)
            
            # Compute circular loss
            loss = criterion(outputs, labels_normalized)
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / max(batch_count, 1)
