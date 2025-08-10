#!/usr/bin/env python3
"""
Script para entrenamiento de modelos de Deep Learning
Extracción de plantas ultrasonido vs ambiente

Incluye:
- CNN 1D para formas de onda
- CNN 2D para espectrogramas  
- ResNet para análisis avanzado
- LSTM/GRU para modelado temporal
- Modelo híbrido CNN-LSTM
- Ensemble de múltiples modelos
- Análisis por canales individuales
- Comparación exhaustiva y optimización
- CLI para selección de modelos
- Guardado automático de checkpoints
- Serialización completa de modelos y resultados
"""

# Configurar variables de entorno antes de importar otras librerías
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
import pickle
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

# === GESTIÓN INTELIGENTE DE GPUs ===
def get_gpu_info():
    """Obtener información detallada de GPUs disponibles"""
    if not torch.cuda.is_available():
        return None, []
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        try:
            # Información básica
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            # Memoria actual
            torch.cuda.set_device(i)
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            gpu_memory_free = gpu_memory_total - gpu_memory_reserved
            
            # Utilización (estimada basada en memoria)
            gpu_utilization = (gpu_memory_reserved / gpu_memory_total) * 100
            
            gpu_info.append({
                'id': i,
                'name': gpu_name,
                'memory_total': gpu_memory_total,
                'memory_allocated': gpu_memory_allocated,
                'memory_reserved': gpu_memory_reserved,
                'memory_free': gpu_memory_free,
                'utilization': gpu_utilization
            })
            
        except Exception as e:
            print(f"⚠️  Error obteniendo info de GPU {i}: {e}")
            
    return gpu_count, gpu_info

def select_optimal_device(prefer_multi_gpu=True):
    """Seleccionar dispositivo óptimo automáticamente"""
    if not torch.cuda.is_available():
        print("🔧 No hay GPU disponible, usando CPU")
        return torch.device('cpu'), False, []
    
    gpu_count, gpu_info = get_gpu_info()
    
    if not gpu_info:
        print("🔧 No se pudo obtener información de GPU, usando CPU")
        return torch.device('cpu'), False, []
    
    print(f"\n🔍 GPUS DETECTADAS: {gpu_count}")
    print("="*50)
    
    for gpu in gpu_info:
        status = "🟢 Libre" if gpu['utilization'] < 20 else "🟡 Ocupada" if gpu['utilization'] < 70 else "🔴 Muy ocupada"
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"   Memoria: {gpu['memory_free']:.1f}GB libre / {gpu['memory_total']:.1f}GB total")
        print(f"   Utilización: {gpu['utilization']:.1f}% {status}")
    
    # Estrategia de selección
    if gpu_count == 1:
        device = torch.device(f'cuda:{gpu_info[0]["id"]}')
        use_multi_gpu = False
        print(f"\n🎯 Usando GPU única: {gpu_info[0]['name']}")
        
    elif prefer_multi_gpu and gpu_count > 1:
        # Verificar si hay suficientes GPUs libres para multi-GPU
        free_gpus = [gpu for gpu in gpu_info if gpu['utilization'] < 50 and gpu['memory_free'] > 2.0]
        
        if len(free_gpus) >= 2:
            device = torch.device('cuda:0')  # GPU principal
            use_multi_gpu = True
            gpu_ids = [gpu['id'] for gpu in free_gpus]
            print(f"\n🚀 Modo Multi-GPU activado!")
            print(f"   GPUs seleccionadas: {gpu_ids}")
            print(f"   GPU principal: {free_gpus[0]['name']}")
        else:
            # Seleccionar la GPU menos cargada
            best_gpu = min(gpu_info, key=lambda x: x['utilization'])
            device = torch.device(f'cuda:{best_gpu["id"]}')
            use_multi_gpu = False
            print(f"\n🎯 GPU menos cargada: GPU {best_gpu['id']} ({best_gpu['name']})")
    else:
        # Seleccionar la GPU menos cargada
        best_gpu = min(gpu_info, key=lambda x: x['utilization'])
        device = torch.device(f'cuda:{best_gpu["id"]}')
        use_multi_gpu = False
        print(f"\n🎯 GPU menos cargada: GPU {best_gpu['id']} ({best_gpu['name']})")
    
    return device, use_multi_gpu, gpu_info

def optimize_batch_size(base_batch_size, device, use_multi_gpu=False):
    """Optimizar batch size basado en GPU disponible"""
    if device.type == 'cpu':
        # CPU: usar batch size conservador
        return min(base_batch_size, 8)
    
    if not torch.cuda.is_available():
        return base_batch_size
    
    try:
        # Obtener memoria GPU disponible
        if use_multi_gpu:
            # Para multi-GPU, usar memoria de la GPU con menos memoria
            min_memory = min([torch.cuda.get_device_properties(i).total_memory 
                             for i in range(torch.cuda.device_count())]) / 1024**3
            total_memory = min_memory
        else:
            current_device = device.index if hasattr(device, 'index') else 0
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        # Sugerir batch size basado en memoria
        if total_memory >= 24:      # RTX 3090/4090, A100
            suggested_batch = 64
        elif total_memory >= 12:    # RTX 3080 Ti, RTX 4070 Ti
            suggested_batch = 32
        elif total_memory >= 8:     # RTX 3070, RTX 4060 Ti
            suggested_batch = 24
        elif total_memory >= 6:     # RTX 3060, GTX 1660
            suggested_batch = 16
        else:                       # GPU con poca memoria
            suggested_batch = 8
        
        # Si multi-GPU, puede manejar batch sizes más grandes
        if use_multi_gpu:
            suggested_batch = min(suggested_batch * 2, 128)
        
        optimized_batch = min(max(base_batch_size, suggested_batch), 128)
        
        if optimized_batch != base_batch_size:
            print(f"💡 Batch size optimizado: {base_batch_size} → {optimized_batch} (Memoria GPU: {total_memory:.1f}GB)")
        
        return optimized_batch
        
    except Exception as e:
        print(f"⚠️  Error optimizando batch size: {e}")
        return base_batch_size

def setup_model_for_device(model, device, use_multi_gpu=False):
    """Configurar modelo para dispositivo óptimo"""
    model = model.to(device)
    
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"🔄 Configurando DataParallel en {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
        # Verificar memoria disponible
        total_memory = sum([torch.cuda.get_device_properties(i).total_memory 
                           for i in range(torch.cuda.device_count())]) / 1024**3
        print(f"💾 Memoria GPU total disponible: {total_memory:.1f}GB")
    
    return model

def monitor_gpu_usage():
    """Monitorear uso de GPU durante entrenamiento"""
    if not torch.cuda.is_available():
        return "CPU"
    
    try:
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        return f"GPU {current_device}: {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%)"
    except:
        return "GPU (error)"

# Configuración de dispositivo inteligente
device, use_multi_gpu, gpu_info = select_optimal_device()
print(f"🔧 Dispositivo seleccionado: {device}")
if use_multi_gpu:
    print(f"🚀 Multi-GPU habilitado")

# Función auxiliar para sampling estratificado
def stratified_sample(df, n, random_state=None, stratify_col='label'):
    """
    Función auxiliar para hacer sampling estratificado manualmente
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples = []
    for label in df[stratify_col].unique():
        label_df = df[df[stratify_col] == label]
        n_samples = max(1, int(n * len(label_df) / len(df)))
        if len(label_df) >= n_samples:
            sample = label_df.sample(n=n_samples, random_state=random_state)
        else:
            sample = label_df
        samples.append(sample)
    
    return pd.concat(samples, ignore_index=True)

# === MODELO CNN 1D ===
class CNN1D(nn.Module):
    """CNN 1D optimizada para análisis de formas de onda de audio"""
    
    def __init__(self, input_length=110250, num_classes=2, dropout=0.3):
        super(CNN1D, self).__init__()
        
        # Capas convolucionales con batch normalization
        self.conv_blocks = nn.Sequential(
            # Bloque 1
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Bloque 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Bloque 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Bloque 4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Dataset para formas de onda
class AudioWaveformDataset(Dataset):
    def __init__(self, dataframe, duration=10.0, sr=22050, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.duration = duration
        self.sr = sr
        self.expected_length = int(duration * sr)
        self.augment = augment
        
    def __len__(self):
        return len(self.dataframe)
    
    def _augment_waveform(self, y):
        """Data augmentation para formas de onda"""
        if self.augment and np.random.random() < 0.5:
            # Time shifting
            shift_samples = int(0.1 * len(y))
            shift = np.random.randint(-shift_samples, shift_samples)
            y = np.roll(y, shift)
            
        if self.augment and np.random.random() < 0.3:
            # Volume augmentation
            volume_factor = np.random.uniform(0.8, 1.2)
            y = y * volume_factor
            
        if self.augment and np.random.random() < 0.3:
            # Gaussian noise
            noise_factor = 0.005
            noise = np.random.normal(0, noise_factor, len(y))
            y = y + noise
            
        return y
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['full_path']
        label = row['label']
        
        try:
            # Cargar audio
            y, _ = librosa.load(audio_path, duration=self.duration, sr=self.sr)
            
            if len(y) == 0:
                return torch.zeros(1, self.expected_length), torch.LongTensor([0]).squeeze()
            
            # Padding o truncating para longitud consistente
            if len(y) < self.expected_length:
                y = np.pad(y, (0, self.expected_length - len(y)), mode='constant')
            else:
                y = y[:self.expected_length]
            
            # Augmentation
            if self.augment:
                y = self._augment_waveform(y)
            
            # Normalización
            y = y / (np.max(np.abs(y)) + 1e-8)
            
            # Convertir a tensor
            waveform_tensor = torch.FloatTensor(y).unsqueeze(0)
            label_tensor = torch.LongTensor([label])
            
            return waveform_tensor, label_tensor.squeeze()
            
        except Exception as e:
            print(f"Error cargando {audio_path}: {e}")
            return torch.zeros(1, self.expected_length), torch.LongTensor([0]).squeeze()

# === MODELO CNN 2D ===
class CNN2D(nn.Module):
    """CNN 2D para análisis de espectrogramas"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(CNN2D, self).__init__()
        
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            # Bloque 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloque 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# === BLOQUE RESIDUAL ===
class ResidualBlock(nn.Module):
    """Bloque residual para ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class AudioResNet(nn.Module):
    """ResNet adaptada para análisis de audio"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(AudioResNet, self).__init__()
        
        # Capa inicial
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Pooling global y clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Dataset para espectrogramas
class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, duration=10.0, sr=22050, n_mels=128, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.duration = duration
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment
        
    def __len__(self):
        return len(self.dataframe)
    
    def _augment_spectrogram(self, spec):
        """Data augmentation para espectrogramas"""
        if self.augment and np.random.random() < 0.3:
            # Frequency masking
            freq_mask_param = max(1, int(0.1 * spec.shape[0]))
            freq_mask = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, spec.shape[0] - freq_mask))
            spec[f0:f0+freq_mask, :] = spec.mean()
            
        if self.augment and np.random.random() < 0.3:
            # Time masking
            time_mask_param = max(1, int(0.1 * spec.shape[1]))
            time_mask = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, spec.shape[1] - time_mask))
            spec[:, t0:t0+time_mask] = spec.mean()
            
        return spec
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['full_path']
        label = row['label']
        
        try:
            # Cargar audio
            y, _ = librosa.load(audio_path, duration=self.duration, sr=self.sr)
            
            if len(y) == 0:
                # Crear espectrograma de tamaño fijo para casos de error
                expected_time_frames = int(self.duration * self.sr / 512) + 1
                return torch.zeros(1, self.n_mels, expected_time_frames), torch.LongTensor([0]).squeeze()
            
            # Asegurar duración mínima
            min_length = int(self.duration * self.sr)
            if len(y) < min_length:
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
            else:
                y = y[:min_length]
            
            # Crear mel-espectrograma con parámetros fijos
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels, 
                n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            
            # Asegurar tamaño fijo en dimensión temporal
            expected_time_frames = int(self.duration * self.sr / 512) + 1
            if mel_spec_db.shape[1] < expected_time_frames:
                # Padding temporal
                pad_width = expected_time_frames - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_spec_db.min())
            elif mel_spec_db.shape[1] > expected_time_frames:
                # Truncar
                mel_spec_db = mel_spec_db[:, :expected_time_frames]
            
            # Augmentation
            if self.augment:
                mel_spec_db = self._augment_spectrogram(mel_spec_db)
            
            # Normalizar
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Convertir a tensor
            spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            label_tensor = torch.LongTensor([label])
            
            return spec_tensor, label_tensor.squeeze()
            
        except Exception as e:
            print(f"Error cargando {audio_path}: {e}")
            # Retornar tensor de tamaño fijo en caso de error
            expected_time_frames = int(self.duration * self.sr / 512) + 1
            return torch.zeros(1, self.n_mels, expected_time_frames), torch.LongTensor([0]).squeeze()

# === MODELOS LSTM ===
class AudioLSTM(nn.Module):
    """LSTM para modelado temporal de características de audio"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(AudioLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Extractor de características CNN
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, input_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Extraer características
        features = self.feature_extractor(x)  # (batch, channels, seq_len)
        features = features.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Clasificar
        output = self.classifier(attended)
        return output

class AudioGRU(nn.Module):
    """GRU alternativo para modelado temporal"""
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(AudioGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Extractor de características
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, input_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU()
        )
        
        # GRU bidireccional
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Clasificador con skip connection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Extraer características
        features = self.feature_extractor(x)
        features = features.permute(0, 2, 1)
        
        # GRU
        gru_out, h_n = self.gru(features)
        
        # Usar última salida de cada dirección
        last_output = gru_out[:, -1, :]
        
        # Clasificar
        output = self.classifier(last_output)
        return output

class HybridCNN_LSTM(nn.Module):
    """Modelo híbrido CNN-LSTM para máximo rendimiento"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(HybridCNN_LSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Bloque 1
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Bloque 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            
            # Bloque 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        
        # LSTM para modelado temporal
        self.lstm = nn.LSTM(
            256, 128, 2, batch_first=True, 
            dropout=dropout, bidirectional=True
        )
        
        # Attention y clasificación
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # CNN features
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq, features)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Clasificar
        output = self.classifier(attended)
        return output

# === FUNCIONES DE ENTRENAMIENTO ===
def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, model_name):
    """Guardar checkpoint durante el entrenamiento"""
    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Cargar checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def convert_for_json(obj):
    """Convertir objetos para serialización JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

def save_model_results(model, results, model_name, results_dir):
    """Guardar modelo y sus resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelo PyTorch
    model_path = results_dir / f"{model_name}_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Guardar modelo completo (para reproducibilidad)
    full_model_path = results_dir / f"{model_name}_full_model_{timestamp}.pt"
    torch.save(model, full_model_path)
    
    # Preparar resultados para JSON usando función de conversión
    results_json = {}
    for k, v in results.items():
        if k == 'model':
            # No guardar el modelo en JSON
            continue
        else:
            results_json[k] = convert_for_json(v)
    
    # Guardar resultados como JSON
    results_path = results_dir / f"{model_name}_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Guardar métricas como CSV para análisis
    metrics_df = pd.DataFrame({
        'Metric': ['accuracy', 'auc', 'best_val_acc'],
        'Value': [results.get('accuracy', 0), results.get('auc', 0), results.get('best_val_acc', 0)]
    })
    metrics_path = results_dir / f"{model_name}_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    return {
        'model_path': model_path,
        'full_model_path': full_model_path,
        'results_path': results_path,
        'metrics_path': metrics_path,
        'timestamp': timestamp
    }

def train_deep_model(model, train_loader, val_loader, num_epochs=50, 
                    learning_rate=1e-3, model_name="Model", device=device,
                    checkpoint_dir=None, save_every=5, start_epoch=0, initial_metrics=None,
                    use_multi_gpu=False):
    """
    Función de entrenamiento optimizada para modelos de deep learning
    con gestión inteligente de GPUs y checkpoints
    """
    # Configurar modelo para dispositivo óptimo
    model = setup_model_for_device(model, device, use_multi_gpu)
    
    # Crear directorio de checkpoints si se especifica
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Función de pérdida con pesos para clases desbalanceadas
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)  # Ajustar si hay desbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizador con weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Métricas - inicializar con valores previos si existen
    if initial_metrics:
        train_losses = initial_metrics.get('train_losses', [])
        val_losses = initial_metrics.get('val_losses', [])
        train_accs = initial_metrics.get('train_accs', [])
        val_accs = initial_metrics.get('val_accs', [])
        best_val_acc = initial_metrics.get('best_val_acc', 0.0)
    else:
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
    
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"🚀 Entrenando {model_name}")
    if start_epoch > 0:
        print(f"   🔄 Continuando desde época {start_epoch+1}")
        print(f"   📊 Mejor accuracy anterior: {best_val_acc:.2f}%")
    print(f"   Épocas: {start_epoch+1}-{num_epochs}, LR: {learning_rate}")
    print(f"   Dispositivo: {device}")
    if use_multi_gpu:
        print(f"   🚀 Multi-GPU: {torch.cuda.device_count()} GPUs")
    print(f"   💾 Memoria GPU: {monitor_gpu_usage()}")
    if checkpoint_dir:
        print(f"   Checkpoints: {checkpoint_dir}")
    
    try:
        for epoch in range(start_epoch, num_epochs):
            # === ENTRENAMIENTO ===
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            train_pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} - Train")
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Métricas
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # Actualizar barra con info GPU
                gpu_info = monitor_gpu_usage()
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%',
                    'GPU': gpu_info
                })
                
                # Limpiar cache GPU cada cierto número de batches
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # === VALIDACIÓN ===
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calcular métricas
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Guardar métricas
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Scheduler step
            scheduler.step()
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Extraer el modelo base si es DataParallel
                model_to_save = model.module if hasattr(model, 'module') else model
                best_model_state = model_to_save.state_dict().copy()
                patience_counter = 0
                print(f"   🎯 Nuevo mejor modelo! Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Guardar checkpoint
            if checkpoint_dir and (epoch + 1) % save_every == 0:
                metrics = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'best_val_acc': best_val_acc
                }
                # Guardar modelo base sin DataParallel wrapper
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_path = save_checkpoint(model_to_save, optimizer, epoch, metrics, checkpoint_dir, model_name)
                print(f"💾 Checkpoint guardado: {checkpoint_path}")
            
            # Log con información GPU
            gpu_status = monitor_gpu_usage()
            print(f"Época {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%, LR {optimizer.param_groups[0]['lr']:.6f}, {gpu_status}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break

    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido en época {epoch+1}")
        if checkpoint_dir:
            metrics = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'best_val_acc': best_val_acc
            }
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = save_checkpoint(model_to_save, optimizer, epoch, metrics, checkpoint_dir, f"{model_name}_interrupted")
            print(f"💾 Checkpoint de emergencia guardado: {checkpoint_path}")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ Error de memoria GPU en época {epoch+1}")
            print(f"   Sugerencias:")
            print(f"   - Reducir batch_size")
            print(f"   - Usar --device cpu")
            print(f"   - Cerrar otros procesos que usen GPU")
            torch.cuda.empty_cache()
        raise e

    # Cargar mejor modelo
    if best_model_state is not None:
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(best_model_state)

    return {
        'model': model.module if hasattr(model, 'module') else model,  # Retornar modelo base
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def evaluate_deep_model(model, test_loader, model_name="Model", device=device):
    """Evaluación completa de modelo de deep learning"""
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluando {model_name}"):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Probabilidades
            probs = F.softmax(output, dim=1)

            _, predicted = torch.max(output, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = 100. * test_correct / test_total

    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except:
        auc_score = 0.0

    print(f"\n📊 {model_name} - Accuracy: {accuracy:.2f}%, AUC: {auc_score:.3f}")
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# === FUNCIONES DE CLI ===
def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de Deep Learning para audio")
    
    # Modelos a entrenar
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn1d', 'cnn2d', 'resnet', 'lstm', 'gru', 'hybrid', 'ensemble', 'all'],
                       default=['all'],
                       help='Modelos a entrenar (default: all)')
    
    # Configuración de datos
    parser.add_argument('--data-path', type=str, default='../data',
                       help='Ruta al directorio de datos')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Tamaño de muestra para prototipado (None para dataset completo)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duración del audio en segundos')
    
    # Configuración de entrenamiento
    parser.add_argument('--epochs', type=int, default=25,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Configuración de checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                       help='Directorio para guardar checkpoints')
    parser.add_argument('--results-dir', type=str, default='../models',
                       help='Directorio para guardar resultados')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Guardar checkpoint cada N épocas')
    
    # Configuración de dispositivo
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                       help='Dispositivo de cómputo (auto=selección inteligente)')
    parser.add_argument('--gpu-strategy', type=str, default='optimal',
                       choices=['optimal', 'single', 'multi', 'least_used'],
                       help='Estrategia de uso de GPU')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Forzar uso de CPU incluso con GPU disponible')
    
    # Modo de ejecución
    parser.add_argument('--resume', type=str, default=None,
                       help='Ruta al checkpoint para continuar entrenamiento')
    parser.add_argument('--resume-from-checkpoints', action='store_true',
                       help='Continuar automáticamente desde los checkpoints más recientes')
    parser.add_argument('--eval-only', action='store_true',
                       help='Solo evaluar modelos existentes')
    parser.add_argument('--compare-only', action='store_true',
                       help='Solo crear comparación final incluyendo algoritmos tradicionales')
    
    return parser.parse_args()

def load_existing_results(results_dir):
    """Cargar resultados existentes"""
    results_dir = Path(results_dir)
    existing_results = {}
    
    # Buscar archivos de resultados
    for results_file in results_dir.glob("*_results_*.json"):
        try:
            model_name = results_file.name.split('_results_')[0]
            with open(results_file, 'r') as f:
                results = json.load(f)
            existing_results[model_name] = results
            print(f"✅ Resultados cargados para {model_name}")
        except Exception as e:
            print(f"⚠️  Error cargando {results_file}: {e}")
    
    return existing_results

def train_model_wrapper(model_name, model_class, train_loader, val_loader, test_loader,
                       config, checkpoint_dir, results_dir):
    """Wrapper para entrenar un modelo específico con manejo de errores"""
    
    print(f"\n{'='*20} ENTRENANDO {model_name.upper()} {'='*20}")
    
    try:
        # Crear modelo
        if model_name == 'cnn1d':
            model = model_class(input_length=int(config['duration']*22050))
        else:
            model = model_class()
        
        # Verificar si hay checkpoint para continuar
        start_epoch = 0
        initial_metrics = None
        model_checkpoint_dir = checkpoint_dir / model_name
        
        if model_checkpoint_dir.exists() and config.get('resume_from_checkpoint', False):
            # Buscar el checkpoint más reciente
            checkpoints = list(model_checkpoint_dir.glob(f"{model_name}_checkpoint_epoch_*.pt"))
            if checkpoints:
                # Obtener el checkpoint con época más alta
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                
                print(f"🔄 Continuando desde checkpoint: {latest_checkpoint}")
                
                # Crear optimizador temporal para cargar estado
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
                
                # Cargar checkpoint
                start_epoch, initial_metrics = load_checkpoint(latest_checkpoint, model, optimizer)
                start_epoch += 1  # Continuar desde la siguiente época
                
                print(f"   📍 Continuando desde época {start_epoch}")
                print(f"   📊 Mejor accuracy anterior: {initial_metrics.get('best_val_acc', 0):.2f}%")
        
        # Entrenar
        results = train_deep_model(
            model, train_loader, val_loader,
            num_epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            model_name=model_name,
            device=config['device'],
            checkpoint_dir=checkpoint_dir / model_name,
            save_every=config['save_every'],
            start_epoch=start_epoch,
            initial_metrics=initial_metrics,
            use_multi_gpu=config.get('use_multi_gpu', False)
        )
        
        # Evaluar
        eval_results = evaluate_deep_model(results['model'], test_loader, model_name, config['device'])
        
        # Combinar resultados
        final_results = {**results, **eval_results}
        
        # Guardar modelo y resultados
        save_info = save_model_results(results['model'], final_results, model_name, results_dir)
        print(f"✅ {model_name} completado y guardado")
        print(f"   Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"   AUC: {eval_results['auc']:.3f}")
        
        return final_results, save_info
        
    except Exception as e:
        print(f"❌ Error entrenando {model_name}: {e}")
        return None, None

# === FUNCIÓN PRINCIPAL ===
def main():
    """Función principal con CLI mejorada"""
    
    # Parsear argumentos
    args = parse_arguments()
    
    print("🎵 SISTEMA DE ENTRENAMIENTO DE DEEP LEARNING")
    print("="*60)
    print(f"Modelos seleccionados: {args.models}")
    print(f"Épocas: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}, Duración: {args.duration}s")
    
    # Configurar dispositivo inteligente
    if args.force_cpu:
        device = torch.device('cpu')
        use_multi_gpu = False
        print("🔧 Forzando uso de CPU")
    elif args.device == 'auto':
        # Usar selección automática inteligente
        prefer_multi = args.gpu_strategy in ['optimal', 'multi']
        device, use_multi_gpu, gpu_info = select_optimal_device(prefer_multi_gpu=prefer_multi)
    elif args.device == 'cpu':
        device = torch.device('cpu')
        use_multi_gpu = False
        print("🔧 Usando CPU")
    elif args.device.startswith('cuda'):
        # GPU específica
        if ':' in args.device:
            gpu_id = int(args.device.split(':')[1])
            if gpu_id < torch.cuda.device_count():
                device = torch.device(args.device)
                use_multi_gpu = False
                print(f"🔧 Usando GPU específica: {args.device}")
            else:
                print(f"⚠️  GPU {gpu_id} no disponible, usando selección automática")
                device, use_multi_gpu, gpu_info = select_optimal_device()
        else:
            device, use_multi_gpu, gpu_info = select_optimal_device()
    else:
        device = torch.device(args.device)
        use_multi_gpu = False
    
    print(f"🎯 Dispositivo final: {device}")
    if use_multi_gpu:
        print(f"🚀 Multi-GPU habilitado con {torch.cuda.device_count()} GPUs")
    
    # Crear directorios
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Si solo evaluación, cargar resultados existentes
    if args.eval_only:
        print("📊 Modo solo evaluación")
        existing_results = load_existing_results(results_dir)
        if existing_results:
            create_final_comparison(existing_results, results_dir)
        else:
            print("❌ No se encontraron resultados existentes")
        return
    
    # Si solo comparación, crear comparación final con todos los resultados
    if args.compare_only:
        print("🏆 Modo solo comparación")
        print("Comparando todos los resultados disponibles (Deep Learning + Traditional ML)")
        
        # Cargar resultados de DL existentes (si los hay)
        existing_dl_results = load_existing_results(results_dir)
        
        # Crear comparación con los resultados disponibles
        create_final_comparison(existing_dl_results, results_dir)
        return
    
    # Cargar dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ No se encuentra directorio de datos: {data_path}")
        return
    
    # Buscar archivos CSV
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        print("❌ No se encuentran archivos CSV en el directorio de datos")
        return
    
    # Cargar el dataset principal
    csv_file = csv_files[0]
    print(f"📂 Cargando dataset: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Dataset cargado: {len(df)} archivos")
        
        # Validar columnas
        required_cols = ['full_path', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ El dataset debe contener las columnas: {required_cols}")
            return
            
        # Filtrar archivos válidos
        df_valid = df[df['full_path'].apply(lambda x: Path(x).exists())].copy()
        print(f"📊 Archivos válidos: {len(df_valid)}")
        
        if len(df_valid) == 0:
            print("❌ No hay archivos de audio válidos")
            return
            
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # Preparar datos
    if args.sample_size:
        df_dl = stratified_sample(df_valid, n=args.sample_size, random_state=42)
        print(f"📊 Usando muestra: {len(df_dl):,} archivos")
    else:
        df_dl = df_valid.copy()
        print(f"📊 Usando dataset completo: {len(df_dl):,} archivos")

    print(f"🎯 Balance: {df_dl['label'].value_counts().to_dict()}")

    # División de datos
    train_df, temp_df = train_test_split(df_dl, test_size=0.3, random_state=42,
                                        stratify=df_dl['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                      stratify=temp_df['label'])

    print(f"\n📈 División:")
    print(f"   Train: {len(train_df):,} ({len(train_df)/len(df_dl)*100:.1f}%)")
    print(f"   Val: {len(val_df):,} ({len(val_df)/len(df_dl)*100:.1f}%)")
    print(f"   Test: {len(test_df):,} ({len(test_df)/len(df_dl)*100:.1f}%)")

    # Configuración
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'duration': args.duration,
        'device': device,
        'save_every': args.save_every,
        'resume_from_checkpoint': args.resume_from_checkpoints,
        'use_multi_gpu': use_multi_gpu,
        'gpu_strategy': args.gpu_strategy
    }
    
    # Definir modelos disponibles
    available_models = {
        'cnn1d': (CNN1D, '1d'),
        'cnn2d': (CNN2D, '2d'),
        'resnet': (AudioResNet, '2d'),
        'lstm': (AudioLSTM, '1d'),
        'gru': (AudioGRU, '1d'),
        'hybrid': (HybridCNN_LSTM, '1d')
    }
    
    # Determinar qué modelos entrenar
    if 'all' in args.models:
        models_to_train = list(available_models.keys())
    else:
        models_to_train = [m for m in args.models if m in available_models]
    
    print(f"\n🎯 Modelos a entrenar: {models_to_train}")
    
    # Optimizar batch size basado en GPU
    original_batch_size = args.batch_size
    optimized_batch_size = optimize_batch_size(original_batch_size, device, use_multi_gpu)
    
    print(f"\n⚙️  CONFIGURACIÓN OPTIMIZADA:")
    print(f"   Batch size: {optimized_batch_size}")
    print(f"   GPU Strategy: {args.gpu_strategy}")
    print(f"   Memoria GPU total: {monitor_gpu_usage()}")
    
    # Crear datasets
    train_dataset_1d = AudioWaveformDataset(train_df, duration=args.duration, augment=True)
    val_dataset_1d = AudioWaveformDataset(val_df, duration=args.duration, augment=False)
    test_dataset_1d = AudioWaveformDataset(test_df, duration=args.duration, augment=False)
    
    train_dataset_2d = SpectrogramDataset(train_df, duration=args.duration, n_mels=128, augment=True)
    val_dataset_2d = SpectrogramDataset(val_df, duration=args.duration, n_mels=128, augment=False)
    test_dataset_2d = SpectrogramDataset(test_df, duration=args.duration, n_mels=128, augment=False)
    
    # DataLoaders con batch size optimizado
    train_loader_1d = DataLoader(train_dataset_1d, batch_size=optimized_batch_size, shuffle=True, num_workers=0)
    val_loader_1d = DataLoader(val_dataset_1d, batch_size=optimized_batch_size, shuffle=False, num_workers=0)
    test_loader_1d = DataLoader(test_dataset_1d, batch_size=optimized_batch_size, shuffle=False, num_workers=0)
    
    train_loader_2d = DataLoader(train_dataset_2d, batch_size=optimized_batch_size, shuffle=True, num_workers=0)
    val_loader_2d = DataLoader(val_dataset_2d, batch_size=optimized_batch_size, shuffle=False, num_workers=0)
    test_loader_2d = DataLoader(test_dataset_2d, batch_size=optimized_batch_size, shuffle=False, num_workers=0)
    
    # Entrenar modelos
    all_results = {}
    all_save_info = {}
    
    for model_name in models_to_train:
        model_class, data_type = available_models[model_name]
        
        # Seleccionar dataloaders apropiados
        if data_type == '1d':
            train_loader, val_loader, test_loader = train_loader_1d, val_loader_1d, test_loader_1d
        else:
            train_loader, val_loader, test_loader = train_loader_2d, val_loader_2d, test_loader_2d
        
        # Entrenar modelo
        results, save_info = train_model_wrapper(
            model_name, model_class, train_loader, val_loader, test_loader,
            config, checkpoint_dir, results_dir
        )
        
        if results:
            all_results[model_name] = results
            all_save_info[model_name] = save_info
    
    # Crear ensemble si se entrenaron múltiples modelos
    if 'ensemble' in args.models and len(all_results) > 1:
        print(f"\n🎭 Creando ensemble...")
        ensemble_results = create_ensemble(all_results, results_dir)
        if ensemble_results:
            all_results['ensemble'] = ensemble_results
    
    # Comparación final
    if all_results:
        create_final_comparison(all_results, results_dir)
        
        # Guardar resumen completo
        summary = {
            'config': config,
            'gpu_info': {
                'device': str(device),
                'multi_gpu': use_multi_gpu,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'batch_size_optimized': optimized_batch_size,
                'batch_size_original': original_batch_size
            },
            'models_trained': list(all_results.keys()),
            'results_summary': {name: {'accuracy': res['accuracy'], 'auc': res['auc']} 
                              for name, res in all_results.items()},
            'save_info': all_save_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convertir summary para JSON
        summary_json = convert_for_json(summary)
        
        summary_path = results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        print(f"\n💾 Resumen completo guardado en: {summary_path}")
    
    print("\n✅ ENTRENAMIENTO COMPLETADO")

def create_ensemble(all_results, results_dir):
    """Crear ensemble de modelos entrenados"""
    try:
        print("🎭 Creando ensemble de modelos...")
        
        # Obtener probabilidades y targets
        all_probs = []
        all_targets = None
        model_names = []
        
        for name, results in all_results.items():
            if 'probabilities' in results and 'targets' in results:
                all_probs.append(results['probabilities'])
                model_names.append(name)
                if all_targets is None:
                    all_targets = results['targets']
        
        if len(all_probs) < 2:
            print("⚠️  Necesitan al menos 2 modelos para ensemble")
            return None
        
        # Pesos basados en AUC
        aucs = [all_results[name]['auc'] for name in model_names]
        weights = np.array(aucs) / np.sum(aucs) if np.sum(aucs) > 0 else None
        
        # Crear ensemble por promedio ponderado
        if weights is not None:
            ensemble_probs = np.zeros_like(all_probs[0])
            for i, probs in enumerate(all_probs):
                ensemble_probs += weights[i] * np.array(probs)
        else:
            ensemble_probs = np.mean(all_probs, axis=0)
        
        # Predicciones
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        # Calcular métricas
        ensemble_accuracy = 100. * np.sum(ensemble_preds == all_targets) / len(all_targets)
        try:
            ensemble_auc = roc_auc_score(all_targets, ensemble_probs)
        except:
            ensemble_auc = 0.0
        
        # Crear resultados del ensemble
        ensemble_results = {
            'accuracy': ensemble_accuracy,
            'auc': ensemble_auc,
            'predictions': ensemble_preds,
            'targets': all_targets,
            'probabilities': ensemble_probs,
            'model_names': model_names,
            'weights': weights if weights is not None else None
        }
        
        # Guardar ensemble
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_path = results_dir / f"ensemble_results_{timestamp}.json"
        
        # Usar función de conversión JSON
        ensemble_json = convert_for_json(ensemble_results)
        
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_json, f, indent=2)
        
        print(f"✅ Ensemble creado:")
        print(f"   Modelos: {model_names}")
        print(f"   Accuracy: {ensemble_accuracy:.2f}%")
        print(f"   AUC: {ensemble_auc:.3f}")
        
        return ensemble_results
        
    except Exception as e:
        print(f"❌ Error creando ensemble: {e}")
        return None

def load_traditional_ml_results(results_dir):
    """Cargar resultados de algoritmos tradicionales de ML"""
    traditional_results = {}
    results_dir = Path(results_dir)
    
    # Buscar archivos CSV con resultados de algoritmos tradicionales
    csv_files = list(results_dir.glob("*training_results*.csv"))
    
    for csv_file in csv_files:
        try:
            print(f"   📂 Cargando: {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            # Verificar que tenga las columnas necesarias
            required_cols = ['Model', 'Accuracy', 'AUC']
            if all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    model_name = row['Model']
                    # Convertir accuracy a porcentaje si está en decimal
                    accuracy = row['Accuracy']
                    if accuracy <= 1.0:
                        accuracy = accuracy * 100
                    
                    traditional_results[model_name] = {
                        'accuracy': accuracy,
                        'auc': row['AUC']
                    }
                    
                print(f"   ✅ Cargados {len(df)} modelos tradicionales")
            else:
                print(f"   ⚠️  Archivo {csv_file.name} no tiene las columnas requeridas: {required_cols}")
                
        except Exception as e:
            print(f"   ❌ Error cargando {csv_file.name}: {e}")
    
    if traditional_results:
        print(f"✅ Total modelos tradicionales cargados: {len(traditional_results)}")
    else:
        print("⚠️  No se encontraron resultados de algoritmos tradicionales")
    
    return traditional_results

def create_final_comparison(all_results, results_dir):
    """Crear comparación final de todos los modelos"""
    print("\n🏆 COMPARACIÓN FINAL DE MODELOS")
    print("="*60)
    
    # Crear DataFrame de comparación con modelos de Deep Learning
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            'Model': f"{model_name} (DL)",
            'Type': 'Deep Learning',
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'Score': metrics['auc'] * 0.6 + (metrics['accuracy']/100) * 0.4
        })
    
    # Buscar y cargar resultados de algoritmos tradicionales
    print("🔍 Buscando resultados de algoritmos tradicionales...")
    traditional_results = load_traditional_ml_results(results_dir)
    
    # Agregar algoritmos tradicionales a la comparación
    for model_name, metrics in traditional_results.items():
        comparison_data.append({
            'Model': f"{model_name} (ML)",
            'Type': 'Traditional ML',
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'Score': metrics['auc'] * 0.6 + (metrics['accuracy']/100) * 0.4
        })
    
    final_comparison_df = pd.DataFrame(comparison_data)
    final_comparison_df = final_comparison_df.sort_values('Score', ascending=False)
    
    print("📊 RANKING FINAL DE MODELOS:")
    print("="*80)
    
    # Mostrar ranking completo
    print("\n🏅 RANKING GENERAL (Todos los modelos):")
    print(final_comparison_df[['Model', 'Type', 'Accuracy', 'AUC', 'Score']].round(3).to_string(index=False))
    
    # Mostrar mejores por categoría
    if len(final_comparison_df) > 0:
        print("\n🎯 MEJORES POR CATEGORÍA:")
        
        # Mejor modelo de Deep Learning
        dl_models = final_comparison_df[final_comparison_df['Type'] == 'Deep Learning']
        if not dl_models.empty:
            best_dl = dl_models.iloc[0]
            print(f"   🤖 Mejor Deep Learning: {best_dl['Model']}")
            print(f"      Accuracy: {best_dl['Accuracy']:.2f}%, AUC: {best_dl['AUC']:.3f}, Score: {best_dl['Score']:.3f}")
        
        # Mejor modelo tradicional
        ml_models = final_comparison_df[final_comparison_df['Type'] == 'Traditional ML']
        if not ml_models.empty:
            best_ml = ml_models.iloc[0]
            print(f"   📊 Mejor Traditional ML: {best_ml['Model']}")
            print(f"      Accuracy: {best_ml['Accuracy']:.2f}%, AUC: {best_ml['AUC']:.3f}, Score: {best_ml['Score']:.3f}")
    
    # Mejor modelo global
    best_model_global = final_comparison_df.iloc[0]
    print(f"\n🥇 CAMPEÓN ABSOLUTO: {best_model_global['Model']}")
    print(f"   Tipo: {best_model_global['Type']}")
    print(f"   Accuracy: {best_model_global['Accuracy']:.2f}%")
    print(f"   AUC: {best_model_global['AUC']:.3f}")
    print(f"   Score: {best_model_global['Score']:.3f}")
    
    # Estadísticas por tipo
    if len(final_comparison_df) > 1:
        print(f"\n📈 ESTADÍSTICAS POR TIPO:")
        stats_by_type = final_comparison_df.groupby('Type')[['Accuracy', 'AUC', 'Score']].agg(['mean', 'max', 'min', 'std'])
        print(stats_by_type.round(3))
    
    # Guardar comparación
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = results_dir / f"final_comparison_{timestamp}.csv"
    final_comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n💾 Comparación guardada en: {comparison_path}")
    
    return final_comparison_df

if __name__ == "__main__":
    main()
