"""
Data loader and preprocessing utilities for plant ultrasonic audio classification
"""
import os
import pandas as pd
import numpy as np
import cv2
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlantAudioDataset(Dataset):
    """Dataset class for plant ultrasonic audio data"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 audio_paths: Optional[List[str]] = None,
                 mode: str = 'train',
                 image_size: Tuple[int, int] = (224, 224),
                 use_audio: bool = False):
        """
        Args:
            image_paths: List of paths to spectrogram images
            labels: List of binary labels (0, 1)
            audio_paths: List of paths to audio files (optional)
            mode: 'train', 'val', or 'test'
            image_size: Target image size for spectrograms
            use_audio: Whether to load audio features
        """
        self.image_paths = image_paths
        self.labels = labels
        self.audio_paths = audio_paths
        self.mode = mode
        self.image_size = image_size
        self.use_audio = use_audio
        
        # Image augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load spectrogram image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transformed = self.transform(image=image)
        image = transformed['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.use_audio and self.audio_paths:
            # Load audio features
            audio_features = self._extract_audio_features(self.audio_paths[idx])
            return {
                'image': image,
                'audio': audio_features,
                'label': label
            }
        
        return {
            'image': image,
            'label': label
        }
    
    def _extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """Extract advanced audio features from WAV file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=30.0)
            
            # Extract multiple features
            features = []
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features.extend([
                np.mean(tonnetz, axis=1),
                np.std(tonnetz, axis=1)
            ])
            
            # Flatten all features
            feature_vector = np.concatenate([np.array(f).flatten() for f in features])
            
            return torch.tensor(feature_vector, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zero vector if loading fails
            return torch.zeros(200, dtype=torch.float32)


class DataLoader:
    """Data loading and preprocessing utilities"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        self.classification_file = self.data_path / "PUA.02" / "classification.tsv"
        self.metadata_file = self.data_path / "PUA.01" / "metadata.tsv"
        self.images_path = self.data_path / "PUA.02" / "images"
        self.audio_path = self.data_path / "PUA.02" / "audiofiles"
        
    def load_classification_data(self) -> pd.DataFrame:
        """Load classification labels"""
        df = pd.read_csv(self.classification_file, sep='\t', header=None, 
                        names=['filename', 'label'])
        return df
    
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata information"""
        df = pd.read_csv(self.metadata_file, sep='\t')
        return df
    
    def find_matching_images(self, classification_df: pd.DataFrame) -> pd.DataFrame:
        """Find matching spectrogram images for classification data"""
        image_data = []
        
        for _, row in classification_df.iterrows():
            filename = row['filename']
            label = row['label']
            
            # Try to find matching image file
            # The filename format might need adjustment based on actual structure
            potential_paths = []
            
            # Search in all subdirectories
            for device_dir in self.images_path.iterdir():
                if device_dir.is_dir():
                    for date_dir in device_dir.iterdir():
                        if date_dir.is_dir():
                            # Look for matching images
                            pattern = filename.replace('/', '_').replace(':', '-')
                            for img_file in date_dir.glob(f"*{pattern}*.jpg"):
                                potential_paths.append(img_file)
            
            if potential_paths:
                # Take the first match
                image_data.append({
                    'filename': filename,
                    'label': label,
                    'image_path': str(potential_paths[0]),
                    'audio_path': None  # Will be filled later
                })
        
        return pd.DataFrame(image_data)
    
    def create_train_val_split(self, 
                              df: pd.DataFrame, 
                              test_size: float = 0.2, 
                              val_size: float = 0.1,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation/test splits"""
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], 
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, 
            stratify=train_val_df['label'], random_state=random_state
        )
        
        return train_df, val_df, test_df
    
    def get_data_loaders(self, 
                        batch_size: int = 32,
                        num_workers: int = 4,
                        use_audio: bool = False) -> Dict[str, DataLoader]:
        """Create PyTorch data loaders for training"""
        
        # Load and prepare data
        classification_df = self.load_classification_data()
        data_df = self.find_matching_images(classification_df)
        
        # Remove rows without matching images
        data_df = data_df.dropna(subset=['image_path'])
        
        print(f"Total samples with matching images: {len(data_df)}")
        print(f"Class distribution:\n{data_df['label'].value_counts()}")
        
        # Create train/val/test splits
        train_df, val_df, test_df = self.create_train_val_split(data_df)
        
        # Create datasets
        datasets = {}
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            dataset = PlantAudioDataset(
                image_paths=split_df['image_path'].tolist(),
                labels=split_df['label'].tolist(),
                audio_paths=split_df['audio_path'].tolist() if use_audio else None,
                mode=split_name,
                use_audio=use_audio
            )
            datasets[split_name] = dataset
        
        # Create data loaders
        data_loaders = {}
        for split_name, dataset in datasets.items():
            shuffle = (split_name == 'train')
            data_loaders[split_name] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers, pin_memory=True
            )
        
        return data_loaders
    
    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution and dataset statistics"""
        df = self.load_classification_data()
        
        stats = {
            'total_samples': len(df),
            'class_distribution': df['label'].value_counts().to_dict(),
            'class_percentages': (df['label'].value_counts(normalize=True) * 100).to_dict(),
            'is_balanced': abs(df['label'].value_counts()[0] - df['label'].value_counts()[1]) / len(df) < 0.1
        }
        
        return stats


def create_stratified_folds(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """Create stratified K-fold splits for cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        folds.append({
            'fold': fold_idx,
            'train': train_df,
            'val': val_df
        })
    
    return folds


if __name__ == "__main__":
    # Example usage
    data_path = "./data"
    loader = DataLoader(data_path)
    
    # Analyze dataset
    stats = loader.analyze_class_distribution()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create data loaders
    data_loaders = loader.get_data_loaders(batch_size=32, use_audio=False)
    
    print(f"\nData loader sizes:")
    for split, dl in data_loaders.items():
        print(f"{split}: {len(dl.dataset)} samples")
