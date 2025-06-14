import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Union, List, Tuple
import os


class GenericDataset(Dataset):
    """
    A generic dataset class that can handle various types of data inputs.
    Supports both classification and regression tasks.
    """

    def __init__(self, data: Union[np.ndarray, List, torch.Tensor],
                 targets: Union[np.ndarray, List, torch.Tensor],
                 transform=None):
        """
        Args:
            data: Input features (can be any type that can be converted to tensor)
            targets: Target labels/values
            transform: Optional transform to be applied on samples
        """
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
        self.targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        self.transform = transform

        # Validate shapes
        if len(self.data) != len(self.targets):
            raise ValueError("Data and targets must have the same length")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


class Trainer:
    """
    Advanced training pipeline for any type of deep learning model and data.
    Features:
    - Supports both classification and regression
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Mixed precision training
    - Model checkpointing
    - Detailed logging
    - Multiple metric tracking
    - Automatic device detection
    - Task type detection (classification/regression)
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[str] = None,
                 task_type: str = 'auto',
                 model_save_path: str = 'checkpoints'):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: Optional DataLoader for test data
            optimizer: Optimizer to use (default: AdamW)
            criterion: Loss function (default: CrossEntropyLoss for classification, MSELoss for regression)
            device: Device to use ('cuda', 'mps', 'cpu') or None for auto-detection
            task_type: 'classification', 'regression', or 'auto' (detect from targets)
            model_save_path: Path to save model checkpoints
        """
        # Device setup
        self.device = device or self._detect_device()
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Task type detection
        self.task_type = self._detect_task_type(task_type)

        # Default optimizer if not provided
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

        # Default loss function based on task type
        self.criterion = criterion or (
            nn.CrossEntropyLoss() if self.task_type == 'classification'
            else nn.MSELoss()
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = 5

        # Model saving
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

        # Additional metrics based on task type
        self.metric_names = (
            ['accuracy', 'precision', 'recall', 'f1'] if self.task_type == 'classification'
            else ['mse', 'mae', 'r2']
        )

    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _detect_task_type(self, task_type: str) -> str:
        """Detect task type if set to 'auto'."""
        if task_type != 'auto':
            return task_type

        # Check first batch to determine task type
        sample_batch, sample_targets = next(iter(self.train_loader))

        # Classification if targets are integers/labels
        if sample_targets.dtype in (torch.long, torch.int):
            return 'classification'
        # Regression if targets are continuous
        elif sample_targets.dtype in (torch.float16, torch.float32, torch.float64):
            return 'regression'
        else:
            raise ValueError("Could not automatically determine task type. Please specify explicitly.")

    def _compute_metrics(self, outputs: torch.Tensor,
                         targets: torch.Tensor) -> Dict[str, float]:
        """Compute metrics based on task type."""
        metrics = {}

        if self.task_type == 'classification':
            _, preds = torch.max(outputs, 1)
            correct = (preds == targets).sum().item()
            total = targets.size(0)
            metrics['accuracy'] = correct / total

            # Additional classification metrics can be added here

        else:  # regression
            mse = nn.functional.mse_loss(outputs.squeeze(), targets).item()
            mae = nn.functional.l1_loss(outputs.squeeze(), targets).item()

            metrics['mse'] = mse
            metrics['mae'] = mae

            # R2 score calculation
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            ss_res = torch.sum((targets - outputs.squeeze()) ** 2)
            r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
            metrics['r2'] = r2

        return metrics

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_outputs = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass and optimize with gradient scaling for mixed precision
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            running_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
            })

        # Concatenate all outputs and targets for metric calculation
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = self._compute_metrics(all_outputs, all_targets)

        return epoch_loss, epoch_metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[Validation]", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)

                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                })

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Calculate validation metrics
        val_loss = running_loss / len(self.val_loader)
        val_metrics = self._compute_metrics(all_outputs, all_targets)

        return val_loss, val_metrics

    def train(self, epochs: int) -> Dict[str, List]:
        """Train the model for the specified number of epochs."""
        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch statistics
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Print metrics
            print("\nTraining Metrics:")
            for name, value in train_metrics.items():
                print(f"{name.capitalize()}: {value:.4f}")

            print("\nValidation Metrics:")
            for name, value in val_metrics.items():
                print(f"{name.capitalize()}: {value:.4f}")

            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print("\nEarly stopping triggered!")
                    break

            # Plot training progress
            self.plot_progress()

        # Load best model
        self.load_model(os.path.join(self.model_save_path, 'best_model.pth'))

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        return self.history

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model on the given loader or test set."""
        if loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided during initialization or evaluation")
            loader = self.test_loader

        self.model.eval()
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="[Evaluating]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Calculate final metrics
        final_loss = self.criterion(all_outputs, all_targets).item()
        final_metrics = self._compute_metrics(all_outputs, all_targets)

        # Print evaluation results
        print("\nEvaluation Results:")
        print(f"Loss: {final_loss:.4f}")
        for name, value in final_metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")

        # Additional reporting for classification tasks
        if self.task_type == 'classification':
            print("\nClassification Report:")
            print(classification_report(
                all_targets.numpy(),
                all_outputs.argmax(dim=1).numpy(),
                digits=4
            ))

            # Confusion matrix
            cm = confusion_matrix(
                all_targets.numpy(),
                all_outputs.argmax(dim=1).numpy()
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        return {
            'loss': final_loss,
            'metrics': final_metrics,
            'outputs': all_outputs.numpy(),
            'targets': all_targets.numpy()
        }

    def save_model(self, filename: str) -> None:
        """Save the model state to a file."""
        path = os.path.join(self.model_save_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'epoch': len(self.history['train_loss'])
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, filename: str) -> None:
        """Load the model state from a file."""
        path = os.path.join(self.model_save_path, filename)
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Model loaded from {path}")

    def plot_progress(self) -> None:
        """Plot training and validation progress."""
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Metrics plot (first metric)
        if self.history['train_metrics']:
            first_metric = next(iter(self.history['train_metrics'][0]))

            plt.subplot(1, 2, 2)
            train_metrics = [m[first_metric] for m in self.history['train_metrics']]
            val_metrics = [m[first_metric] for m in self.history['val_metrics']]

            plt.plot(train_metrics, label=f'Train {first_metric.capitalize()}')
            plt.plot(val_metrics, label=f'Val {first_metric.capitalize()}')
            plt.title(f'{first_metric.capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(first_metric.capitalize())
            plt.legend()

        plt.tight_layout()
        plt.show()