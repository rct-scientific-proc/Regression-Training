import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from normalize import normalize_label

class ImageRegressionDataset(Dataset):
    """Dataset for loading images and their regression labels."""
    
    def __init__(self, csv_file, num_channels=3, transform=None, normalize_mean=None, normalize_std=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
                           CSV should have the image path in the first column and
                           the label value in the second column, with headers.
            num_channels (int): Number of channels for output images (1 for grayscale, 3 for RGB).
            transform (callable, optional): Optional transform to be applied on images.
            normalize_mean (tuple, optional): Mean for each RGB channel for z-score normalization.
                                             If None, only min-max scaling (0-1) is applied.
                                             Example: (0.485, 0.456, 0.406) for ImageNet.
            normalize_std (tuple, optional): Std dev for each RGB channel for z-score normalization.
                                            Must be provided if normalize_mean is provided.
                                            Example: (0.229, 0.224, 0.225) for ImageNet.
        """
        self.data_df = pd.read_csv(csv_file)
        self.num_channels = num_channels
        
        # Dynamically get column names from the CSV
        columns = self.data_df.columns.tolist()
        if len(columns) < 2:
            raise ValueError(f"CSV must have at least 2 columns, but got {len(columns)}")
        
        # First column is image path, second column is label
        self.img_col = columns[0]
        self.label_col = columns[1]
        
        self.transform = transform
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Validate that both mean and std are provided together
        if (normalize_mean is None) != (normalize_std is None):
            raise ValueError("Both normalize_mean and normalize_std must be provided together, or both None")
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a float tensor.
        """
        # Get absolute path from first column (dynamically determined)
        img_path = self.data_df.iloc[idx][self.img_col]
        
        # Load image from file
        image = Image.open(img_path)

        # Handle channel conversion based on num_channels
        if self.num_channels == 1:
            # Convert to grayscale for single channel
            image = image.convert('L')  # 'L' mode is grayscale
        else:
            # Ensure image is in RGB format for 3 channels
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Apply transform to PIL Image if provided (e.g., resizing)
        if self.transform:
            image = self.transform(image)

        # Convert to numpy array
        image_np = np.array(image, dtype=np.float32)
        
        # Ensure 3D array (H, W, C) even for grayscale
        if image_np.ndim == 2:
            image_np = image_np[:, :, np.newaxis]  # Add channel dimension for grayscale
        
        # Normalize to [0, 1] range
        # Standard images are 0-255, but handle potential 16-bit images (0-65535)
        max_val = 255.0 if image_np.max() <= 255 else 65535.0
        image_np = image_np / max_val
        
        # Apply z-score normalization if mean and std are provided
        if self.normalize_mean is not None and self.normalize_std is not None:
            # Subtract mean and divide by std for each channel
            # normalize_mean and normalize_std should have length matching num_channels
            mean = np.array(self.normalize_mean, dtype=np.float32).reshape(1, 1, len(self.normalize_mean))
            std = np.array(self.normalize_std, dtype=np.float32).reshape(1, 1, len(self.normalize_std))
            image_np = (image_np - mean) / std
        
        # Transpose to PyTorch format (C, H, W)
        image_np = np.transpose(image_np, (2, 0, 1))
        
        # Convert to tensor
        image = torch.from_numpy(image_np)
        
        # Get label from second column (dynamically determined)
        label = self.data_df.iloc[idx][self.label_col]
        
        # Normalize label using the normalize_label function
        label_array = np.array([label]).reshape(-1, 1)
        normalized_label = normalize_label(label_array).flatten()[0]
        
        label = torch.tensor(normalized_label, dtype=torch.float32)            
        return image, label
