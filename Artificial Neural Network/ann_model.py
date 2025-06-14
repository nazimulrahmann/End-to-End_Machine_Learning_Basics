import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtificialNN(nn.Module):
    """
    A flexible and reusable Artificial Neural Network with customizable hidden layers,
    batch normalization, dropout, and weight initialization.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layers: list = [1024, 512, 256],
        dropout_rates: list = [0.5, 0.3, 0.2],
        use_batchnorm: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        weight_init: str = "kaiming"
    ):
        """
        Parameters:
            input_size (int): Number of input features
            num_classes (int): Number of output classes
            hidden_layers (list): Number of units per hidden layer
            dropout_rates (list): Dropout rate per hidden layer
            use_batchnorm (bool): Whether to use BatchNorm
            activation_fn (nn.Module): Activation function class (e.g., nn.ReLU)
            weight_init (str): Initialization method - 'kaiming', 'xavier', or 'none'
        """
        super(ArtificialNN, self).__init__()

        assert len(dropout_rates) == len(hidden_layers), "Mismatch in layers and dropout lengths."

        self.flatten = nn.Flatten()

        # Dynamically build hidden layers
        layers = []
        prev_size = input_size
        for idx, layer_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, layer_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(activation_fn())
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(dropout_rates[idx]))
            prev_size = layer_size

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            activation_fn(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[-1] // 2, num_classes)
        )

        # Weight initialization
        self._initialize_weights(weight_init)

    def forward(self, x):
        x = self.flatten(x)
        x = self.feature_extractor(x)
        return self.classifier(x)

    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Usage of Artificial Neural Network model
model = ArtificialNN(
    input_size=784,
    num_classes=10,
    hidden_layers=[512, 256],
    dropout_rates=[0.3, 0.2],
    activation_fn=nn.LeakyReLU,
    weight_init='xavier'
)