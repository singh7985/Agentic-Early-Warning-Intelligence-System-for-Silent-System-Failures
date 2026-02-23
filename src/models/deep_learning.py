"""
Deep Learning Models for RUL Prediction

Implements advanced neural network architectures:
- LSTM (Long Short-Term Memory)
- TCN (Temporal Convolutional Network)
- Bidirectional LSTM
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Parameters
        ----------
        X : np.ndarray
            Features (N, seq_len, features) or (N, features)
        y : np.ndarray
            RUL targets (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRULPredictor(nn.Module):
    """LSTM model for RUL prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int, default=64
            Hidden layer size
        num_layers : int, default=2
            Number of LSTM layers
        dropout : float, default=0.2
            Dropout rate
        bidirectional : bool, default=False
            Use bidirectional LSTM
        """
        super(LSTMRULPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, features)

        Returns
        -------
        output : torch.Tensor
            RUL predictions (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take output from last time step
        if self.bidirectional:
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(lstm_out)
        return output.squeeze(-1)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        """Initialize TCN block."""
        super(TCNBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRULPredictor(nn.Module):
    """Temporal Convolutional Network for RUL prediction."""

    def __init__(
        self,
        input_size: int,
        num_channels: list = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize TCN model.

        Parameters
        ----------
        input_size : int
            Number of input features
        num_channels : list, default=[64, 64, 64]
            Channel sizes for each TCN layer
        kernel_size : int, default=3
            Convolution kernel size
        dropout : float, default=0.2
            Dropout rate
        """
        super(TCNRULPredictor, self).__init__()
        
        self.input_size = input_size
        layers = []
        
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, features)

        Returns
        -------
        output : torch.Tensor
            RUL predictions (batch,)
        """
        # Reshape for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN forward
        out = self.network(x)
        
        # Fully connected
        out = self.fc(out)
        return out.squeeze(-1)


class DeepLearningTrainer:
    """Trainer for deep learning models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            PyTorch model
        learning_rate : float, default=0.001
            Learning rate
        device : str, optional
            Device ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized trainer on device: {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 128,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Train model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        epochs : int, default=100
            Number of epochs
        batch_size : int, default=128
            Batch size
        early_stopping_patience : int, default=10
            Early stopping patience
        verbose : bool, default=True
            Print progress

        Returns
        -------
        history : dict
            Training history
        """
        # Ensure 3D shape for sequential models
        if len(X_train.shape) == 2:
            X_train = X_train[:, np.newaxis, :]
            if X_val is not None:
                X_val = X_val[:, np.newaxis, :]
        
        train_dataset = RULDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, batch_size)
                self.val_losses.append(val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        logger.info("✓ Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
        }

    def _validate(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> float:
        """Validate model."""
        self.model.eval()
        val_dataset = RULDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features
        batch_size : int, default=128
            Batch size

        Returns
        -------
        predictions : np.ndarray
            RUL predictions
        """
        # Ensure 3D shape
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]
        
        self.model.eval()
        dataset = RULDataset(X, np.zeros(len(X)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                preds = self.model(batch_X)
                predictions.append(preds.cpu().numpy())
        
        return np.concatenate(predictions)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model.

        Parameters
        ----------
        X : np.ndarray
            Features
        y_true : np.ndarray
            True RUL values

        Returns
        -------
        metrics : dict
            RMSE, MAE, R2, MAPE
        """
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        d = y_pred - y_true
        nasa_score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
        }
        
        logger.info(f"Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, NASA Score={nasa_score:.2f}")
        return metrics

    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")
