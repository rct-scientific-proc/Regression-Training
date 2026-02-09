import numpy as np


def normalize_label(labels: np.ndarray) -> np.ndarray:
    """
    Normalize label values to a range suitable for neural network training.
    
    === WHY NORMALIZE LABELS? ===
    
    Neural networks train best when outputs are in a bounded, predictable range.
    Common target ranges: [0, 1] or [-1, 1]
    
    Benefits:
    - Faster convergence during training
    - Better gradient flow through the network
    - Prevents exploding/vanishing gradients
    - Works well with sigmoid/tanh output activations
    
    === HOW TO MODIFY THIS FUNCTION ===
    
    1. Identify your label's original range (min, max)
    2. Choose your target range (usually [0, 1] or [-1, 1])
    3. Apply the appropriate transformation below
    
    === COMMON NORMALIZATION STRATEGIES ===
    
    # Min-Max Normalization to [0, 1]:
    # normalized = (labels - min_val) / (max_val - min_val)
    # Example: [5, 10] → [0, 1]
    #   normalized = (labels - 5) / (10 - 5)
    
    # Min-Max Normalization to [-1, 1]:
    # normalized = 2 * (labels - min_val) / (max_val - min_val) - 1
    # Example: [0, 100] → [-1, 1]
    #   normalized = 2 * (labels - 0) / (100 - 0) - 1
    
    # Z-Score Normalization (mean=0, std=1):
    # normalized = (labels - mean) / std
    # Use when you don't know the min/max bounds
    
    # Logarithmic Normalization (for exponential ranges):
    # normalized = np.log(labels + 1) / np.log(max_val + 1)
    # Use for labels spanning several orders of magnitude
    
    === EXAMPLES FOR DIFFERENT USE CASES ===
    
    Example 1: Temperature in Celsius [-20°C to 40°C] → [0, 1]
        shifted = labels + 20      # Shift to [0, 60]
        normalized = shifted / 60   # Scale to [0, 1]
    
    Example 2: Distance in meters [0m to 1000m] → [0, 1]
        normalized = labels / 1000
    
    Example 3: Pixel column index [0 to 255] → [0, 1]
        normalized = labels / 255
    
    Example 4: Percentage [-100% to 100%] → [0, 1]
        normalized = (labels + 100) / 200
    
    Example 5: Age in years [0 to 120] → [0, 1]
        normalized = labels / 120
    
    Example 6: Angle in degrees [0° to 360°] → [0, 1]
        normalized = labels / 360
    
    IMPORTANT: When you modify this function, ensure your transformation is INVERTIBLE
    if you need to convert predictions back to original units later.
    
    Args:
        labels (np.ndarray): Label values to normalize
                            Can be shape (n_samples,) or (n_samples, 1)
    
    Returns:
        np.ndarray: Normalized labels in [0, 1] range
                   Shape will be (n_samples, 1) - column vector
    
    === TO IMPLEMENT YOUR OWN NORMALIZATION ===
    
    Replace the transformation code below with your own logic:
    
    def normalize_label(labels: np.ndarray) -> np.ndarray:
        # Ensure column vector
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        # YOUR NORMALIZATION HERE
        # Example: normalize [0, 100] to [0, 1]
        min_val = 0
        max_val = 100
        normalized = (labels - min_val) / (max_val - min_val)
        
        return normalized
    """
    
    # ========================================================================
    # STEP 1: Ensure labels are in column vector format
    # ========================================================================
    # Reshape from (n_samples,) to (n_samples, 1) if needed
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    
    # ========================================================================
    # STEP 2: Apply normalization transformation
    # ========================================================================
    # Current implementation: Normalize radians [-π/2, π/2] → [0, 1]
    
    # Shift the range from [-π/2, π/2] to [0, π]
    # This makes the minimum value 0
    shifted_labels = labels + 90.0
    
    # Scale to [0, 1] by dividing by the range (180)
    # Final range: [0, 1]
    one_over_180 = 1 / 180.0
    normalized_labels = shifted_labels * one_over_180
    
    # ========================================================================
    # STEP 3: Return normalized labels
    # ========================================================================
    return normalized_labels


# ============================================================================
# QUICK REFERENCE: Common Normalization Functions
# ============================================================================
#
# Copy these into your code and modify as needed
#
# === Generic Min-Max Normalization ===
# def normalize_label(labels, min_val, max_val):
#     if len(labels.shape) == 1:
#         labels = labels.reshape(-1, 1)
#     normalized = (labels - min_val) / (max_val - min_val)
#     return normalized
#
# === Z-Score Normalization ===
# def normalize_label(labels, mean=None, std=None):
#     if len(labels.shape) == 1:
#         labels = labels.reshape(-1, 1)
#     if mean is None:
#         mean = np.mean(labels)
#     if std is None:
#         std = np.std(labels)
#     normalized = (labels - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
#     return normalized
#
# === Log Normalization (for exponential data) ===
# def normalize_label(labels, max_val):
#     if len(labels.shape) == 1:
#         labels = labels.reshape(-1, 1)
#     normalized = np.log(labels + 1) / np.log(max_val + 1)
#     return normalized
#
# === No Normalization (if labels are already in [0, 1]) ===
# def normalize_label(labels):
#     if len(labels.shape) == 1:
#         labels = labels.reshape(-1, 1)
#     return labels  # Return as-is if already normalized
#

