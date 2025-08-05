"""
Training utilities and helper functions
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LearningRateScheduler:
    """Custom learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int = 5,
                 max_epochs: int = 100,
                 warmup_lr: float = 1e-6,
                 base_lr: float = 1e-3,
                 min_lr: float = 1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            warmup_lr: Learning rate at start of warmup
            base_lr: Peak learning rate after warmup
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            cosine_epochs = self.max_epochs - self.warmup_epochs
            current_cosine_epoch = self.current_epoch - self.warmup_epochs
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * current_cosine_epoch / cosine_epochs)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [N, C]
            targets: Ground truth labels [N]
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss for severe class imbalance"""
    
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean'):
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to prevent overconfident predictions"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        return torch.mean(loss)


class MixUp:
    """MixUp data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Apply MixUp augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Apply CutMix augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = y, y[index]
        
        # Get random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size: torch.Size, lam: float) -> tuple:
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class AudioAugmentation:
    """Audio-specific augmentation techniques"""
    
    def __init__(self, 
                 time_stretch_rate: float = 0.1,
                 pitch_shift_steps: int = 2,
                 noise_factor: float = 0.005,
                 time_shift_ms: int = 100):
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_factor = noise_factor
        self.time_shift_ms = time_shift_ms
    
    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply time stretching"""
        import librosa
        stretch_factor = 1 + np.random.uniform(-self.time_stretch_rate, self.time_stretch_rate)
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply pitch shifting"""
        import librosa
        n_steps = np.random.randint(-self.pitch_shift_steps, self.pitch_shift_steps + 1)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random noise"""
        noise = np.random.randn(len(audio))
        return audio + self.noise_factor * noise
    
    def time_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply time shifting"""
        shift_samples = int(self.time_shift_ms * sr / 1000)
        shift = np.random.randint(-shift_samples, shift_samples + 1)
        
        if shift > 0:
            return np.pad(audio[shift:], (0, shift), mode='wrap')
        elif shift < 0:
            return np.pad(audio[:shift], (-shift, 0), mode='wrap')
        else:
            return audio


def get_class_weights(labels: np.ndarray, method: str = 'balanced') -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    if method == 'balanced':
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    elif method == 'sqrt':
        # Square root of inverse frequency
        unique, counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / np.sqrt(counts)
        class_weights = class_weights / class_weights.sum() * len(unique)
    else:
        # Equal weights
        class_weights = np.ones(len(np.unique(labels)))
    
    return torch.FloatTensor(class_weights)


def create_advanced_augmentations(image_size: tuple = (224, 224), mode: str = 'train') -> A.Compose:
    """Create advanced image augmentations"""
    
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            ], p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_losses'], label='Training Loss')
    axes[0, 0].plot(history['val_losses'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[0, 1].plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rates' in history:
        axes[1, 0].plot(history['learning_rates'], label='Learning Rate', color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Loss difference (overfitting indicator)
    if 'train_losses' in history and 'val_losses' in history:
        loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
        axes[1, 1].plot(loss_diff, label='Validation - Training Loss', color='red')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_random_seed(seed: int = 42):
    """Alias for set_seed function for backward compatibility"""
    return set_seed(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3)
    
    # Simulate training
    val_losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]
    for i, loss in enumerate(val_losses):
        should_stop = early_stopping(loss)
        print(f"Epoch {i+1}: Loss = {loss}, Should stop = {should_stop}")
        if should_stop:
            break
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    inputs = torch.randn(4, 2)  # 4 samples, 2 classes
    targets = torch.tensor([0, 1, 0, 1])
    loss = focal_loss(inputs, targets)
    print(f"Focal loss: {loss.item()}")
    
    # Test class weights
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1])  # Imbalanced
    weights = get_class_weights(labels)
    print(f"Class weights: {weights}")
    
    print("All utilities working correctly!")
