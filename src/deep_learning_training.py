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
"""

# Configurar variables de entorno antes de importar otras librerías
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
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

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Dispositivo: {device}")

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
                return torch.zeros(1, self.n_mels, 432), torch.LongTensor([0]).squeeze()
            
            # Crear mel-espectrograma
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels, 
                n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            
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
            return torch.zeros(1, self.n_mels, 432), torch.LongTensor([0]).squeeze()

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
def train_deep_model(model, train_loader, val_loader, num_epochs=50, 
                    learning_rate=1e-3, model_name="Model", device=device):
    """
    Función de entrenamiento optimizada para modelos de deep learning
    """
    model = model.to(device)
    
    # Función de pérdida con pesos para clases desbalanceadas
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)  # Ajustar si hay desbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizador con weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Métricas
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"🚀 Entrenando {model_name}")
    print(f"   Épocas: {num_epochs}, LR: {learning_rate}, Dispositivo: {device}")
    
    for epoch in range(num_epochs):
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
            
            # Actualizar barra
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
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
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log
        print(f"Época {epoch+1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%, LR {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
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

# === FUNCIÓN PRINCIPAL ===
def main():
    """Función principal para ejecutar el entrenamiento de deep learning"""
    
    print("🎵 INICIANDO ENTRENAMIENTO DE DEEP LEARNING")
    print("="*60)
    
    # Cargar dataset
    data_path = Path("../data")
    if not data_path.exists():
        print("❌ No se encuentra directorio de datos")
        return
    
    # Buscar archivos CSV
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        print("❌ No se encuentran archivos CSV en el directorio de datos")
        return
    
    # Cargar el dataset principal
    csv_file = csv_files[0]  # Usar el primer CSV encontrado
    print(f"📂 Cargando dataset: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Dataset cargado: {len(df)} archivos")
        
        # Validar que existan las columnas necesarias
        required_cols = ['full_path', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ El dataset debe contener las columnas: {required_cols}")
            return
            
        # Filtrar archivos que existen
        df_valid = df[df['full_path'].apply(lambda x: Path(x).exists())].copy()
        print(f"📊 Archivos válidos: {len(df_valid)}")
        
        if len(df_valid) == 0:
            print("❌ No hay archivos de audio válidos")
            return
            
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return
    
    # Configuración para deep learning
    USE_FULL_DATASET = True  # Cambiar a True para dataset completo
    
    if USE_FULL_DATASET:
        df_dl = df_valid.copy()
    else:
        # Usar muestra estratificada para prototipado
        sample_size = min(1000, len(df_valid))
        df_dl = stratified_sample(df_valid, n=sample_size, random_state=42)

    print(f"📊 Dataset para Deep Learning: {len(df_dl):,} archivos")
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

    # Configuración de entrenamiento
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-3
    DURATION = 5.0  # segundos de audio

    print(f"\n⚙️  Configuración:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Épocas: {NUM_EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Duración audio: {DURATION}s")
    
    # Resultados de modelos
    dl_results = {}
    
    # === ENTRENAR CNN 1D ===
    print("\n🎵 Entrenando CNN 1D...")
    
    # Crear datasets 1D
    train_dataset_1d = AudioWaveformDataset(train_df, duration=DURATION, augment=True)
    val_dataset_1d = AudioWaveformDataset(val_df, duration=DURATION, augment=False)
    test_dataset_1d = AudioWaveformDataset(test_df, duration=DURATION, augment=False)

    # DataLoaders
    train_loader_1d = DataLoader(train_dataset_1d, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_1d = DataLoader(val_dataset_1d, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader_1d = DataLoader(test_dataset_1d, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modelo CNN 1D
    model_cnn1d = CNN1D(input_length=int(DURATION*22050))

    # Entrenar
    results_cnn1d = train_deep_model(
        model_cnn1d, train_loader_1d, val_loader_1d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_name="CNN1D", device=device
    )

    # Evaluar
    eval_cnn1d = evaluate_deep_model(results_cnn1d['model'], test_loader_1d, "CNN1D", device)
    dl_results['CNN1D'] = eval_cnn1d
    
    # === ENTRENAR CNN 2D ===
    print("\n🌈 Entrenando CNN 2D...")

    # Crear datasets 2D
    train_dataset_2d = SpectrogramDataset(train_df, duration=DURATION, n_mels=128, augment=True)
    val_dataset_2d = SpectrogramDataset(val_df, duration=DURATION, n_mels=128, augment=False)
    test_dataset_2d = SpectrogramDataset(test_df, duration=DURATION, n_mels=128, augment=False)

    # DataLoaders
    train_loader_2d = DataLoader(train_dataset_2d, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_2d = DataLoader(val_dataset_2d, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader_2d = DataLoader(test_dataset_2d, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modelo CNN 2D
    model_cnn2d = CNN2D()

    # Entrenar
    results_cnn2d = train_deep_model(
        model_cnn2d, train_loader_2d, val_loader_2d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_name="CNN2D", device=device
    )

    # Evaluar
    eval_cnn2d = evaluate_deep_model(results_cnn2d['model'], test_loader_2d, "CNN2D", device)
    dl_results['CNN2D'] = eval_cnn2d
    
    # === ENTRENAR ResNet ===
    print("\n🏗️  Entrenando AudioResNet...")

    # Modelo ResNet
    model_resnet = AudioResNet()

    # Entrenar (usando mismos dataloaders 2D)
    results_resnet = train_deep_model(
        model_resnet, train_loader_2d, val_loader_2d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE*0.8,  # LR ligeramente menor
        model_name="AudioResNet", device=device
    )

    # Evaluar
    eval_resnet = evaluate_deep_model(results_resnet['model'], test_loader_2d, "AudioResNet", device)
    dl_results['AudioResNet'] = eval_resnet
    
    # === ENTRENAR LSTM ===
    print("\n⏰ Entrenando AudioLSTM...")
    
    # Modelo LSTM
    model_lstm = AudioLSTM()
    
    # Entrenar (usando dataloaders 1D)
    results_lstm = train_deep_model(
        model_lstm, train_loader_1d, val_loader_1d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE*0.5,  # LR menor para LSTM
        model_name="AudioLSTM", device=device
    )
    
    # Evaluar
    eval_lstm = evaluate_deep_model(results_lstm['model'], test_loader_1d, "AudioLSTM", device)
    dl_results['AudioLSTM'] = eval_lstm
    
    # === ENTRENAR GRU ===
    print("\n🔄 Entrenando AudioGRU...")
    
    # Modelo GRU
    model_gru = AudioGRU()
    
    # Entrenar
    results_gru = train_deep_model(
        model_gru, train_loader_1d, val_loader_1d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE*0.5,
        model_name="AudioGRU", device=device
    )
    
    # Evaluar
    eval_gru = evaluate_deep_model(results_gru['model'], test_loader_1d, "AudioGRU", device)
    dl_results['AudioGRU'] = eval_gru
    
    # === ENTRENAR HÍBRIDO CNN-LSTM ===
    print("\n🚀 Entrenando HybridCNN_LSTM...")
    
    # Modelo híbrido
    model_hybrid = HybridCNN_LSTM()
    
    # Entrenar
    results_hybrid = train_deep_model(
        model_hybrid, train_loader_1d, val_loader_1d,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE*0.7,
        model_name="HybridCNN_LSTM", device=device
    )
    
    # Evaluar
    eval_hybrid = evaluate_deep_model(results_hybrid['model'], test_loader_1d, "HybridCNN_LSTM", device)
    dl_results['HybridCNN_LSTM'] = eval_hybrid
    
    # === ENSEMBLE ===
    print("\n🎭 Creando ensemble de modelos...")
    
    # Ensemble simple por votación ponderada
    def ensemble_predict(models_probs, weights=None):
        """Ensemble por promedio ponderado de probabilidades"""
        if weights is None:
            weights = [1.0] * len(models_probs)
        
        # Normalizar pesos
        weights = np.array(weights) / np.sum(weights)
        
        # Promedio ponderado
        ensemble_probs = np.zeros_like(models_probs[0])
        for i, probs in enumerate(models_probs):
            ensemble_probs += weights[i] * np.array(probs)
        
        # Predicciones
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        return ensemble_preds, ensemble_probs
    
    # Obtener probabilidades de todos los modelos
    all_probs = []
    all_targets = None
    model_names = []
    
    for name, results in dl_results.items():
        all_probs.append(results['probabilities'])
        model_names.append(name)
        if all_targets is None:
            all_targets = results['targets']
    
    # Pesos basados en AUC
    aucs = [dl_results[name]['auc'] for name in model_names]
    weights = np.array(aucs) / np.sum(aucs) if np.sum(aucs) > 0 else None
    
    # Crear ensemble
    ensemble_preds, ensemble_probs = ensemble_predict(all_probs, weights)
    
    # Calcular métricas del ensemble
    ensemble_accuracy = 100. * np.sum(ensemble_preds == all_targets) / len(all_targets)
    try:
        ensemble_auc = roc_auc_score(all_targets, ensemble_probs)
    except:
        ensemble_auc = 0.0
    
    # Añadir a resultados
    dl_results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'auc': ensemble_auc,
        'predictions': ensemble_preds,
        'targets': all_targets,
        'probabilities': ensemble_probs
    }
    
    print(f"✅ Ensemble creado:")
    print(f"   Modelos: {model_names}")
    print(f"   Pesos: {weights.round(3) if weights is not None else 'Iguales'}")
    print(f"   Accuracy: {ensemble_accuracy:.2f}%")
    print(f"   AUC: {ensemble_auc:.3f}")
    
    # === COMPARACIÓN FINAL ===
    print("\n🏆 COMPARACIÓN FINAL DE MODELOS")
    print("="*60)
    
    # Crear DataFrame de comparación
    comparison_data = []
    for model_name, metrics in dl_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'Score': metrics['auc'] * 0.6 + (metrics['accuracy']/100) * 0.4  # Score compuesto
        })
    
    final_comparison_df = pd.DataFrame(comparison_data)
    final_comparison_df = final_comparison_df.sort_values('Score', ascending=False)
    
    print("📊 RANKING FINAL DE MODELOS:")
    print(final_comparison_df.round(3))
    
    # Mejor modelo
    best_model_global = final_comparison_df.iloc[0]
    print(f"\n🥇 MEJOR MODELO: {best_model_global['Model']}")
    print(f"   Accuracy: {best_model_global['Accuracy']:.2f}%")
    print(f"   AUC: {best_model_global['AUC']:.3f}")
    print(f"   Score: {best_model_global['Score']:.3f}")
    
    # Guardar resultados
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = models_dir / f"deep_learning_results_{timestamp}.csv"
    final_comparison_df.to_csv(results_path, index=False)
    print(f"\n💾 Resultados guardados en: {results_path}")
    
    print("\n✅ ENTRENAMIENTO DE DEEP LEARNING COMPLETADO")
    print("🚀 Framework completo de deep learning implementado!")

if __name__ == "__main__":
    main()
