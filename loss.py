import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    """
    Custom loss function for regression tasks.
    
    This is a template for implementing custom loss functions that work with PyTorch.
    Users can modify the forward() method to implement their own loss computation.
    
    IMPORTANT: For the loss to work with gradient descent (backpropagation), all operations
    must be differentiable PyTorch operations. Avoid numpy operations or conditional logic
    that isn't supported by PyTorch's autograd system.
    
    === HOW TO IMPLEMENT A CUSTOM LOSS FUNCTION ===
    
    1. Inherit from nn.Module:
        class MyCustomLoss(nn.Module):
            def __init__(self):
                super(MyCustomLoss, self).__init__()
            
            def forward(self, predictions, targets):
                # Your loss computation here
                loss = compute_loss(predictions, targets)
                return loss
    
    2. The forward() method MUST:
        - Accept predictions and targets as tensors
        - Return a scalar loss value (torch.Tensor with single element)
        - Use only differentiable PyTorch operations
    
    3. Supported PyTorch operations for gradients:
        - torch.abs(), torch.mean(), torch.sum()
        - torch.where(), torch.clamp(), torch.exp()
        - torch.pow(), torch.sqrt(), torch.log()
        - Arithmetic: +, -, *, /, **
        - torch.nn.functional operations (F.mse_loss, F.l1_loss, etc.)
    
    4. Avoid non-differentiable operations:
        - numpy operations
        - if/else statements (use torch.where instead)
        - argmax/sorting on continuous values
        - .item() (extracts value from graph)
    
    === EXAMPLE: Huber Loss (L1/L2 hybrid) ===
    
        class HuberLoss(nn.Module):
            def __init__(self, delta=1.0):
                super(HuberLoss, self).__init__()
                self.delta = delta
            
            def forward(self, predictions, targets):
                diff = torch.abs(predictions - targets)
                # Smooth transition between L2 (small errors) and L1 (large errors)
                loss = torch.where(
                    diff <= self.delta,
                    0.5 * diff ** 2,  # L2 for small errors
                    self.delta * (diff - 0.5 * self.delta)  # L1 for large errors
                )
                return torch.mean(loss)
    
    === CURRENT IMPLEMENTATION ===
    
    This class implements a circular loss for periodic/angular regression.
    It's useful when predictions wrap around (e.g., angles 0-2π or 0-1).
    
    The loss accounts for circular distance: the shortest path between two angles
    on a circle. For example, 0.95 and 0.05 (normalized to [0,1]) are actually
    very close when representing angles (separated by 0.1 on the circle).
    """
    
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Compute circular MSE loss for normalized angles in [0, 1].
        
        This loss accounts for wraparound: 0 and 1 are considered close together
        (representing the same angle in different forms).
        
        Args:
            predictions (torch.Tensor): Model predictions, shape (batch_size,) or (batch_size, 1)
                                       Values should be in [0, 1] range
            targets (torch.Tensor): Ground truth labels, shape (batch_size,) or (batch_size, 1)
                                   Values should be in [0, 1] range
        
        Returns:
            torch.Tensor: Scalar loss value (single number)
        
        === HOW THIS LOSS WORKS ===
        
        For each (prediction, target) pair:
        1. Compute linear distance: |pred - target|
        2. Consider wraparound: distance could also be 1 - |pred - target|
        3. Take minimum: min(linear, wraparound)
        4. Square it and average across batch
        
        Example:
            pred = 0.95, target = 0.05
            linear_dist = |0.95 - 0.05| = 0.90
            wraparound_dist = 1 - 0.90 = 0.10
            circular_dist = min(0.90, 0.10) = 0.10  ← These are actually close!
        """
        # Compute error for each sample in batch
        error = predictions - targets
        
        # Compute absolute error
        abs_error = torch.abs(error)
        
        # For circular distance on [0,1], check both directions:
        # - Direct distance: abs_error
        # - Wraparound distance: 1 - abs_error
        # Use torch.where for differentiable conditional
        # (unlike if/else which breaks the gradient graph)
        circular_error = torch.where(
            abs_error <= 0.5,      # If error is small
            abs_error,              # Use it directly
            1.0 - abs_error         # Otherwise use wraparound distance
        )
        
        # Mean Squared Error of circular distance
        # Square the error, then average across batch
        loss = torch.mean(circular_error ** 2)
        
        return loss


# ============================================================================
# QUICK REFERENCE: Other Common Loss Functions
# ============================================================================
#
# To use these, copy them into loss.py and import/use in train.py
#
# === Mean Squared Error (MSE) - Simple L2 distance ===
# loss = torch.mean((predictions - targets) ** 2)
#
# === Mean Absolute Error (MAE) - Simple L1 distance ===
# loss = torch.mean(torch.abs(predictions - targets))
#
# === Smooth L1 Loss - Robust to outliers ===
# loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
#
# === Huber Loss - Customizable robustness ===
# delta = 1.0
# diff = torch.abs(predictions - targets)
# loss = torch.mean(torch.where(
#     diff <= delta,
#     0.5 * diff ** 2,
#     delta * (diff - 0.5 * delta)
# ))
#
# === Percentage Error - For normalized outputs ===
# error = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
# loss = torch.mean(error ** 2)
#
