import torch


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
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Ensure shapes match
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Compute differentiable circular loss
        loss = criterion(outputs, labels)
        
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
            
            # Forward pass
            outputs = model(images).squeeze()
            
            # Ensure shapes match
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Compute circular loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / max(batch_count, 1)
