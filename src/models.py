import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, X):
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        attention_weights = self.softmax(Q @ K.transpose(-2, -1) / self.hidden_dim**0.5)
        attention_out = attention_weights @ V
        attention_out = self.fc(attention_out)
        return attention_out

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        attention_hidden_dim: int = 64  # Hidden dimension for attention
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.attention = AttentionLayer(hid_dim, attention_hidden_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = X.permute(0, 2, 1)  # Permute to (b, t, c) for attention

        residual = X  # Store the input for the residual connection

        X = self.attention(X)
        X = X + residual  # Add the residual connection

        X = X.permute(0, 2, 1)  # Permute back to (b, c, t)
        
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = self.conv0(X)
        X = F.gelu(self.batchnorm0(residual))

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))  # Applying GELU after adding the skip connection

        X = self.conv2(X)
        X = F.gelu(self.batchnorm2(X))  # Replaced GLU with GELU to maintain channel size

        return self.dropout(X)