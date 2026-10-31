"""
Neural Network Architectures for Hand-Washing Recognition (Ivanovs & Ivanova, Baltic DB&IS 2026)
Implements:
1. LSTM (Long Short-Term Memory) - Table 1
2. GRU (Gated Recurrent Unit) - Table 2
3. 1D Temporal CNN (Convolutional Neural Network) - Table 3
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch import nn

from config import DROPOUT_RATE, NUM_CLASSES


class HandLSTM(nn.Module):
    """
    LSTM Sequential Model (Table 1).
    - Recurrent layers: 2 or 3 LSTM layers
    - Hidden size: 64 or 128 units
    - Dropout: 0.2 between layers and before classifier
    - Classifier: Fully connected layer to 7 classes
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = DROPOUT_RATE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_size)
        final_state = lstm_out[:, -1, :]  # Take the final frame representation
        out = self.dropout(final_state)
        logits = self.classifier(out)  # (B, num_classes)
        return logits


class HandGRU(nn.Module):
    """
    GRU Sequential Model (Table 2).
    - Recurrent layers: 1 or 2 GRU layers
    - Hidden size: 128 or 256 units
    - Dropout: 0.2
    - Classifier head: Linear(128) + ReLU + Dropout(0.2) + Linear(7)
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = DROPOUT_RATE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        gru_out, _ = self.gru(x)  # (B, T, hidden_size)
        final_state = gru_out[:, -1, :]  # Final frame hidden state
        logits = self.classifier_head(final_state)
        return logits


class HandTemporalCNN(nn.Module):
    """
    1D Temporal CNN Model (Table 3 & Section 3.6).
    Achieved highest performance in the paper:
    - 2 or 3 Conv1D layers along the time axis
    - Number of filters: 64 or 128
    - Kernel size: 3 or 5
    - Activation: ReLU
    - Dropout: 0.2 after each conv layer
    - Temporal aggregation: Adaptive Average Pooling to 1
    - Classifier head: Linear(filters, 64) + ReLU + Dropout(0.2) + Linear(64, 7)
    """

    def __init__(
        self,
        input_dim: int = 126,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = DROPOUT_RATE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = num_filters

        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Transpose so feature dimension becomes channel dimension: (B, input_dim, seq_len)
        x_trans = x.transpose(1, 2)
        features = self.conv_stack(x_trans)  # (B, num_filters, seq_len)
        pooled = self.global_pool(features).squeeze(-1)  # (B, num_filters)
        logits = self.classifier_head(pooled)  # (B, num_classes)
        return logits


def get_model(model_name: str, input_dim: int = 126, **kwargs) -> nn.Module:
    """Factory helper to instantiate models by name."""
    name = model_name.lower()
    if "cnn" in name or "conv" in name:
        return HandTemporalCNN(
            input_dim=input_dim,
            num_filters=kwargs.get("filters", 64),
            kernel_size=kwargs.get("kernel_size", 3),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", DROPOUT_RATE),
        )
    elif "gru" in name:
        return HandGRU(
            input_dim=input_dim,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", DROPOUT_RATE),
        )
    elif "lstm" in name:
        return HandLSTM(
            input_dim=input_dim,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", DROPOUT_RATE),
        )
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'cnn', 'gru', 'lstm'."
        )
