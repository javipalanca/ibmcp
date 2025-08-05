"""
Advanced deep learning models for plant ultrasonic audio classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoConfig
import timm
from typing import Dict, Optional, Tuple

class SpectrogramCNN(nn.Module):
    """Advanced CNN for spectrogram classification using pre-trained models"""
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        Args:
            model_name: Name of the backbone model (efficientnet, resnet, etc.)
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for regularization
        """
        super(SpectrogramCNN, self).__init__()
        
        # Load pre-trained model
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
            else:
                self.feature_dim = features.shape[-1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling if needed
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        # Classify
        output = self.classifier(features)
        return output


class VisionTransformer(nn.Module):
    """Vision Transformer for spectrogram classification"""
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1):
        super(VisionTransformer, self).__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer (AST) for audio classification"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 model_size: str = 'base',
                 pretrained: bool = True):
        super(AudioSpectrogramTransformer, self).__init__()
        
        # AST configuration
        if model_size == 'base':
            self.config = {
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'patch_size': 16,
                'frequency_stride': 10,
                'time_stride': 10
            }
        elif model_size == 'small':
            self.config = {
                'hidden_size': 384,
                'num_hidden_layers': 12,
                'num_attention_heads': 6,
                'patch_size': 16,
                'frequency_stride': 10,
                'time_stride': 10
            }
        
        # Use ViT as backbone and adapt for audio
        self.ast = timm.create_model(
            'vit_base_patch16_224' if model_size == 'base' else 'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config['hidden_size']),
            nn.Dropout(0.1),
            nn.Linear(self.config['hidden_size'], num_classes)
        )
    
    def forward(self, x):
        # Extract features using transformer
        features = self.ast(x)
        output = self.classifier(features)
        return output


class HybridCNNRNN(nn.Module):
    """Hybrid CNN-RNN model for temporal pattern recognition"""
    
    def __init__(self, 
                 cnn_backbone: str = 'resnet50',
                 rnn_type: str = 'LSTM',
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        super(HybridCNNRNN, self).__init__()
        
        # CNN backbone for feature extraction
        self.cnn = timm.create_model(
            cnn_backbone,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Get CNN feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            cnn_features = self.cnn(dummy_input)
            self.cnn_feature_dim = cnn_features.shape[1]
        
        # Temporal modeling with RNN
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.cnn_feature_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            rnn_output_size = hidden_size * 2  # Bidirectional
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                self.cnn_feature_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            rnn_output_size = hidden_size * 2
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            rnn_output_size, 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Extract spatial features with CNN
        cnn_features = self.cnn(x)  # [B, C, H', W']
        
        # Reshape for RNN: treat spatial dimensions as time steps
        features = cnn_features.view(batch_size, self.cnn_feature_dim, -1)
        features = features.permute(0, 2, 1)  # [B, T, C]
        
        # Temporal modeling with RNN
        rnn_output, _ = self.rnn(features)
        
        # Apply attention
        attended_output, _ = self.attention(rnn_output, rnn_output, rnn_output)
        
        # Global pooling over time dimension
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Classification
        output = self.classifier(pooled_output)
        return output


class MultiModalModel(nn.Module):
    """Multi-modal model combining spectrograms and audio features"""
    
    def __init__(self,
                 spectrogram_model: str = 'efficientnet_b4',
                 audio_feature_dim: int = 200,
                 num_classes: int = 2,
                 fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
                 dropout_rate: float = 0.3):
        super(MultiModalModel, self).__init__()
        
        # Spectrogram branch
        self.spectrogram_cnn = timm.create_model(
            spectrogram_model,
            pretrained=True,
            num_classes=0
        )
        spec_feature_dim = self.spectrogram_cnn.num_features
        
        # Audio feature branch
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            fusion_dim = spec_feature_dim + 256
        elif fusion_method == 'add':
            # Project to same dimension for addition
            self.spec_projection = nn.Linear(spec_feature_dim, 256)
            fusion_dim = 256
        elif fusion_method == 'attention':
            self.attention_fusion = nn.MultiheadAttention(
                256, num_heads=8, dropout=dropout_rate, batch_first=True
            )
            self.spec_projection = nn.Linear(spec_feature_dim, 256)
            fusion_dim = 256
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, spectrogram, audio_features):
        # Process spectrogram
        spec_features = self.spectrogram_cnn(spectrogram)
        
        # Process audio features
        audio_features = self.audio_processor(audio_features)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused_features = torch.cat([spec_features, audio_features], dim=1)
        elif self.fusion_method == 'add':
            spec_features = self.spec_projection(spec_features)
            fused_features = spec_features + audio_features
        elif self.fusion_method == 'attention':
            spec_features = self.spec_projection(spec_features)
            # Prepare for attention: [B, 1, D] for both modalities
            spec_features = spec_features.unsqueeze(1)
            audio_features = audio_features.unsqueeze(1)
            
            # Cross-modal attention
            attended_features, _ = self.attention_fusion(
                spec_features, audio_features, audio_features
            )
            fused_features = attended_features.squeeze(1)
        
        # Classification
        output = self.classifier(fused_features)
        return output


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models: Dict[str, nn.Module], ensemble_method: str = 'voting'):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleDict(models)
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'stacking':
            # Meta-learner for stacking
            input_dim = len(models) * 2  # Assuming 2 classes
            self.meta_learner = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
    
    def forward(self, *inputs):
        predictions = []
        
        for name, model in self.models.items():
            if isinstance(inputs[0], dict):
                # Multi-modal input
                pred = model(**inputs[0])
            else:
                # Single input
                pred = model(inputs[0])
            predictions.append(pred)
        
        if self.ensemble_method == 'voting':
            # Average predictions
            ensemble_output = torch.stack(predictions, dim=0).mean(dim=0)
        elif self.ensemble_method == 'stacking':
            # Use meta-learner
            stacked_predictions = torch.cat(predictions, dim=1)
            ensemble_output = self.meta_learner(stacked_predictions)
        
        return ensemble_output


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models"""
    
    if model_type == 'cnn':
        return SpectrogramCNN(**kwargs)
    elif model_type == 'vit':
        return VisionTransformer(**kwargs)
    elif model_type == 'ast':
        return AudioSpectrogramTransformer(**kwargs)
    elif model_type == 'hybrid':
        return HybridCNNRNN(**kwargs)
    elif model_type == 'multimodal':
        return MultiModalModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CNN model
    model = create_model('cnn', model_name='efficientnet_b4')
    model.to(device)
    
    # Test input
    x = torch.randn(4, 3, 224, 224).to(device)
    output = model(x)
    print(f"CNN output shape: {output.shape}")
    
    # Test Vision Transformer
    vit_model = create_model('vit')
    vit_model.to(device)
    vit_output = vit_model(x)
    print(f"ViT output shape: {vit_output.shape}")
    
    print("All models created successfully!")
