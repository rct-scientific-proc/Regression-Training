import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for regression with configurable layers."""
    
    def __init__(self, input_channels=3, image_size=256, dropout_rate=0.3, min_feature_size=4, num_conv_layers=4, num_fc_layers=2):
        super(SimpleCNN, self).__init__()
        
        # Build convolutional layers dynamically
        conv_layers = nn.Sequential()
        in_channels = input_channels
        out_channels = 32
        
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            
            # Use AdaptiveAvgPool2d for last layer, MaxPool2d for others
            if i == num_conv_layers - 1:
                conv_layers.append(nn.AdaptiveAvgPool2d((min_feature_size, min_feature_size)))
            else:
                conv_layers.append(nn.MaxPool2d(2))
            
            in_channels = out_channels
            # Double channels up to 128, then keep at 128
            if out_channels < 128:
                out_channels = min(out_channels * 2, 128)
        
        self.conv_layers = conv_layers
        
        # Calculate flattened size based on min_feature_size and final channel count
        self.final_channels = in_channels
        self.flattened_size = self.final_channels * min_feature_size * min_feature_size
        
        # Build fully connected layers dynamically
        fc_layers = nn.Sequential()
        fc_layers.append(nn.Flatten())
        
        # Create hidden layers with geometric progression of sizes
        if num_fc_layers > 0:
            layer_sizes = []
            # Generate decreasing sizes from flattened_size to intermediate values
            current_size = self.flattened_size
            for i in range(num_fc_layers):
                # Calculate next size: geometric decrease
                next_size = max(64, int(current_size / 2))
                layer_sizes.append((current_size, next_size))
                current_size = next_size
            
            # Add hidden layers
            for i, (in_size, out_size) in enumerate(layer_sizes):
                fc_layers.append(nn.Linear(in_size, out_size))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
            
            # Add output layer from last hidden layer
            fc_layers.append(nn.Linear(layer_sizes[-1][1], 1))
        else:
            # If no hidden layers, go directly from flattened to output
            fc_layers.append(nn.Linear(self.flattened_size, 1))
        
        fc_layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.fc_layers = fc_layers
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
