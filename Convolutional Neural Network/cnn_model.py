import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNN(nn.Module):
    """
    A flexible and reusable deep convolutional neural network with configurable architecture.
    Supports both 1D (time series) and 2D (image) data, with optional residual connections.
    """

    def __init__(self, input_shape, num_classes, conv_layers=None, fc_layers=None,
                 use_batchnorm=True, use_dropout=True, dropout_rate=0.5,
                 activation='relu', pooling='max', residual=False):
        """
        Initialize a configurable CNN model.

        Args:
            input_shape (tuple): Shape of input data (channels, height, width) or (channels, length) for 1D
            num_classes (int): Number of output classes
            conv_layers (list): List of tuples (out_channels, kernel_size, stride, padding)
                               If None, uses default architecture
            fc_layers (list): List of integers specifying hidden units in fully connected layers
            use_batchnorm (bool): Whether to use batch normalization
            use_dropout (bool): Whether to use dropout
            dropout_rate (float): Dropout probability
            activation (str): Activation function ('relu', 'leakyrelu', 'elu', 'selu')
            pooling (str): Pooling type ('max', 'avg', 'adaptive')
            residual (bool): Whether to use residual connections
        """
        super(ConvolutionalNN, self).__init__()

        # Validate input
        if len(input_shape) not in [2, 3]:
            raise ValueError("input_shape must be (channels, height, width) or (channels, length)")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.is_1d = len(input_shape) == 2

        # Set default architecture if not provided
        if conv_layers is None:
            if self.is_1d:
                conv_layers = [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)]
            else:
                conv_layers = [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)]

        if fc_layers is None:
            fc_layers = [1024, 512]

        # Select layer types based on input dimension
        conv_layer = nn.Conv1d if self.is_1d else nn.Conv2d
        bn_layer = nn.BatchNorm1d if self.is_1d else nn.BatchNorm2d
        pool_layer = nn.MaxPool1d if self.is_1d else nn.MaxPool2d
        adaptive_pool_layer = nn.AdaptiveAvgPool1d if self.is_1d else nn.AdaptiveAvgPool2d

        # Configure activation
        self.activation = self._get_activation(activation)

        # Configure pooling
        self.pooling_type = pooling
        if pooling == 'adaptive':
            self.adaptive_pool_output = (4,) if self.is_1d else (4, 4)

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = input_shape[0]

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            conv_block = nn.ModuleDict()

            # Convolutional layer
            conv_block['conv'] = conv_layer(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

            # Batch normalization
            if use_batchnorm:
                conv_block['bn'] = bn_layer(out_channels)

            # Pooling (not in last layer if using adaptive pooling)
            if pooling == 'adaptive' and i < len(conv_layers) - 1:
                conv_block['pool'] = pool_layer(kernel_size=2, stride=2)
            elif pooling != 'adaptive':
                conv_block['pool'] = pool_layer(kernel_size=2, stride=2)

            self.conv_blocks.append(conv_block)
            in_channels = out_channels

        # Adaptive pooling if selected
        if pooling == 'adaptive':
            self.adaptive_pool = adaptive_pool_layer(self.adaptive_pool_output)

        # Calculate flattened size after conv layers
        with torch.no_grad():
            self._flattened_size = self._get_conv_output(input_shape)

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self._flattened_size

        for units in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, units))
            in_features = units

        # Final output layer
        self.output_layer = nn.Linear(in_features, num_classes)

        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation):
        """Configure activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leakyrelu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _get_conv_output(self, shape):
        """Calculate the output size after convolutional layers"""
        dummy_input = torch.zeros(1, *shape)
        x = dummy_input

        for block in self.conv_blocks:
            x = block['conv'](x)
            if 'bn' in block:
                x = block['bn'](x)
            x = self.activation(x)
            if 'pool' in block:
                x = block['pool'](x)

        if self.pooling_type == 'adaptive':
            x = self.adaptive_pool(x)

        return x.flatten().shape[0]

    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional blocks
        for i, block in enumerate(self.conv_blocks):
            residual = x if self.residual and i > 0 and x.shape[1] == block['conv'].out_channels else None

            x = block['conv'](x)
            if 'bn' in block:
                x = block['bn'](x)
            x = self.activation(x)
            if 'pool' in block:
                x = block['pool'](x)

            if residual is not None:
                x += residual

        # Adaptive pooling if selected
        if self.pooling_type == 'adaptive':
            x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
            x = self.activation(x)
            if self.use_dropout:
                x = self.dropout(x)

        # Output layer (no activation)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        """Initialize weights with appropriate methods"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_summary(self):
        """Return a summary of the model architecture"""
        summary = []
        summary.append(f"Input Shape: {self.input_shape}")
        summary.append("Convolutional Blocks:")

        for i, block in enumerate(self.conv_blocks):
            conv = block['conv']
            layer_type = "Conv1D" if self.is_1d else "Conv2D"
            summary.append(f"  Block {i + 1}: {layer_type}(in={conv.in_channels}, out={conv.out_channels}, "
                           f"kernel={conv.kernel_size}, stride={conv.stride}, padding={conv.padding})")
            if 'bn' in block:
                summary.append("    BatchNorm")
            if 'pool' in block:
                pool = block['pool']
                summary.append(f"    {type(pool).__name__}(kernel_size={pool.kernel_size}, stride={pool.stride})")

        if self.pooling_type == 'adaptive':
            summary.append(f"AdaptivePool(output_size={self.adaptive_pool_output})")

        summary.append(f"Flattened Size: {self._flattened_size}")
        summary.append("Fully Connected Layers:")

        for i, layer in enumerate(self.fc_layers):
            summary.append(f"  FC {i + 1}: {layer.in_features} -> {layer.out_features}")

        summary.append(f"Output Layer: {self.fc_layers[-1].out_features} -> {self.num_classes}")

        if self.use_dropout:
            summary.append(f"Dropout: p={self.dropout_rate}")

        summary.append(f"Activation: {self.activation._get_name()}")
        summary.append(f"Residual Connections: {self.residual}")

        return "\n".join(summary)


    # Usage for Convolutional Neural Network
    # For image classification
    model = ConvolutionalNN(
        input_shape=(3, 32, 32),  # 3-channel 32x32 images
        num_classes=10,
        conv_layers=[(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)],
        fc_layers=[512, 256],
        activation='leakyrelu',
        pooling='adaptive',
        residual=True
    )

    # For time series classification
    model = ConvolutionalNN(
        input_shape=(1, 128),  # 1-channel 128-length sequences
        num_classes=5,
        conv_layers=[(16, 5, 1, 2), (32, 5, 1, 2), (64, 3, 1, 1)],
        fc_layers=[256, 128],
        activation='relu',
        pooling='max'
    )

    # Print model summary
    print(model.get_summary())
