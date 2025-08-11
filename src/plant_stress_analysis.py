#!/usr/bin/env python3
"""
TAREA 2: Análisis de Ultrasonidos de Plantas para Predicción de Estado Hídrico

Este script analiza específicamente los ultrasonidos identificados como emisiones de plantas
para predecir su estado de riego y estrés hídrico. Implementa múltiples enfoques de Deep Learning
y análisis temporal para entender las "quejas" ultrasónicas de las plantas.

Objetivos:
- Predecir tiempo desde último riego (last_watering)
- Clasificar estado hídrico (bien regada vs estresada)
- Analizar diferencias entre especies/genotipos
- Detectar patrones temporales de estrés
- Crear sistema de alerta temprana de necesidad de riego

Funcionalidades principales:
- Análisis exploratorio de datos (--analyze-only)
- Entrenamiento de modelos de Deep Learning
- Visualización comparativa de resultados (--visualize-results)
- Evaluación de múltiples arquitecturas (CNN, LSTM, GRU, Transformer, Ensemble)

Uso:
  # Análisis exploratorio
  python plant_stress_analysis.py --analyze-only
  
  # Entrenar modelos
  python plant_stress_analysis.py --models all --epochs 50
  
  # Visualizar resultados comparativos
  python plant_stress_analysis.py --visualize-results

Autor: Equipo IBMCP
Fecha: Agosto 2025
"""

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings('ignore')

# === CONFIGURACIÓN GLOBAL ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Dispositivo: {device}")

# Configurar múltiples GPUs si están disponibles
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🎮 GPUs disponibles: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Usar DataParallel si hay múltiples GPUs
    use_multi_gpu = num_gpus > 1
else:
    use_multi_gpu = False

# === FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS ===

def load_metadata_and_classification(data_path):
    """Cargar y combinar metadata con clasificaciones"""
    data_path = Path(data_path)
    
    all_metadata = []
    
    # Cargar metadata de ambas sesiones
    for session_dir in ['PUA.01', 'PUA.02']:
        session_path = data_path / session_dir
        if not session_path.exists():
            continue
            
        print(f"📂 Procesando {session_dir}...")
        
        # Cargar metadata
        metadata_file = session_path / 'metadata.tsv'
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file, sep='\t')
            metadata['session_dir'] = session_dir
            all_metadata.append(metadata)
    
    # Combinar metadata
    if all_metadata:
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        print(f"✅ Metadata cargada: {len(combined_metadata)} registros")
    else:
        combined_metadata = pd.DataFrame()
    
    # Cargar clasificaciones desde el archivo principal
    classification_file = data_path / 'dataset_combined.csv'
    print(f"🔍 Buscando clasificaciones en: {classification_file}")
    if classification_file.exists():
        classifications_df = pd.read_csv(classification_file)
        print(f"✅ Clasificaciones cargadas: {len(classifications_df)} archivos")
    else:
        classifications_df = pd.DataFrame()
        print("❌ No se encontró archivo de clasificaciones")
    
    return combined_metadata, classifications_df

def parse_datetime(datetime_str):
    """Parsear formato de fecha/hora del dataset"""
    if pd.isna(datetime_str) or datetime_str == '':
        return None
    try:
        return datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S.%f')
        except ValueError:
            return None

def create_plant_ultrasound_dataset(data_path, metadata, classifications):
    """Crear dataset específico de ultrasonidos de plantas"""
    data_path = Path(data_path)
    plant_sounds = []
    
    print("🌱 Creando dataset de ultrasonidos de plantas...")
    
    # Filtrar solo sonidos clasificados como ultrasonidos de plantas
    plant_audio_df = classifications[classifications['label_text'] == 'Ultrasonido_Planta']
    print(f"📊 Ultrasonidos de plantas encontrados: {len(plant_audio_df)}")
    
    for idx, audio_row in tqdm(plant_audio_df.iterrows(), desc="Procesando ultrasonidos", total=len(plant_audio_df)):
        # Obtener información del archivo
        filename = audio_row['filename']
        session_folder = audio_row['session']
        batch = audio_row['batch']
        datetime_info = audio_row['datetime']
        
        # Extraer canal del filename
        if 'ch' in filename and 'PUA_' in filename:
            channel = filename.split('PUA_')[0]  # ch1, ch2, etc.
        else:
            continue
        
        # Buscar metadata correspondiente - filtrar NaN también
        matching_metadata = metadata[
            (metadata['session'] == session_folder) & 
            (metadata['channel'] == channel) &
            (metadata['plant'].notna()) &  # Filtrar plantas NaN
            (metadata['last_watering'].notna())  # Filtrar riegos NaN
        ]
        
        if matching_metadata.empty:
            continue
        
        # Usar el primer match (debería ser único)
        meta_row = matching_metadata.iloc[0]
        
        # Parsear fechas
        recording_time = pd.to_datetime(datetime_info)
        last_watering_time = parse_datetime(meta_row['last_watering'])
        
        if last_watering_time is None:
            continue
        
        # Calcular tiempo desde último riego
        time_since_watering = recording_time - last_watering_time
        hours_since_watering = time_since_watering.total_seconds() / 3600
        
        # Construir ruta del archivo de audio
        session_dir = batch  # Usar directamente el batch (PUA.01 o PUA.02)
        audio_path = data_path / session_dir / 'audiofiles' / session_folder / filename
        
        # Verificar que el archivo existe
        if not audio_path.exists():
            # Intentar con formato alternativo
            alt_audio_path = data_path / session_dir / 'audiofiles' / (filename + '.wav')
            if alt_audio_path.exists():
                audio_path = alt_audio_path
            else:
                continue
        
        # Agregar al dataset
        plant_sounds.append({
            'audio_path': str(audio_path),
            'session': session_folder,
            'channel': channel,
            'plant_id': meta_row['plant'],
            'species': meta_row['species'],
            'genotype': meta_row['genotype'],
            'recording_time': recording_time,
            'last_watering_time': last_watering_time,
            'hours_since_watering': hours_since_watering,
            'days_since_watering': hours_since_watering / 24,
            'session_dir': session_dir,
            'treatment': meta_row['treatment'],
            'soil_sensors': meta_row['soil_sensors'],
            'sowing': meta_row['sowing'],
            'transplant': meta_row['transplant']
        })
    
    plant_df = pd.DataFrame(plant_sounds)
    
    if not plant_df.empty:
        # Filtrar archivos que realmente existen
        existing_files = plant_df[plant_df['audio_path'].apply(lambda x: Path(x).exists())]
        print(f"✅ Dataset creado: {len(existing_files)} ultrasonidos de plantas con archivos existentes")
        
        # Estadísticas básicas
        print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
        print(f"   Especies únicas: {existing_files['species'].nunique()}")
        print(f"   Plantas únicas: {existing_files['plant_id'].nunique()}")
        print(f"   Canales: {sorted(existing_files['channel'].unique())}")
        print(f"   Rango temporal: {existing_files['hours_since_watering'].min():.1f}h - {existing_files['hours_since_watering'].max():.1f}h")
        print(f"   Días desde riego: {existing_files['days_since_watering'].min():.1f} - {existing_files['days_since_watering'].max():.1f}")
        
        return existing_files
    else:
        print("❌ No se encontraron ultrasonidos de plantas válidos")
        return pd.DataFrame()

def create_water_stress_labels(df, stress_threshold_hours=72):
    """Crear etiquetas de estrés hídrico"""
    df = df.copy()
    
    # Etiqueta binaria: 0 = bien regada, 1 = estresada
    df['water_stress'] = (df['hours_since_watering'] > stress_threshold_hours).astype(int)
    
    # Etiquetas categóricas más detalladas
    def categorize_water_status(hours):
        if hours <= 24:
            return 'fresh'        # Recién regada
        elif hours <= 48:
            return 'adequate'     # Riego adecuado
        elif hours <= 72:
            return 'moderate'     # Estrés moderado
        else:
            return 'stressed'     # Estresada
    
    df['water_status'] = df['hours_since_watering'].apply(categorize_water_status)
    
    # Normalizar valores objetivo para regresión
    df['hours_normalized'] = df['hours_since_watering'] / df['hours_since_watering'].max()
    df['days_normalized'] = df['days_since_watering'] / df['days_since_watering'].max()
    
    print(f"\n🎯 DISTRIBUCIÓN DE ESTRÉS HÍDRICO:")
    print(f"   Bien regadas (< {stress_threshold_hours}h): {(df['water_stress'] == 0).sum()}")
    print(f"   Estresadas (≥ {stress_threshold_hours}h): {(df['water_stress'] == 1).sum()}")
    print(f"\n📊 Distribución por categorías:")
    print(df['water_status'].value_counts())
    
    return df

# === MODELOS DE DEEP LEARNING PARA ANÁLISIS DE PLANTAS ===

class PlantStressDataset(Dataset):
    """Dataset para análisis de estrés hídrico en plantas"""
    
    def __init__(self, dataframe, target_type='regression', target_column='hours_since_watering', 
                 duration=5.0, sr=22050, augment=False, representation='spectrogram'):
        """
        Args:
            target_type: 'regression' para predecir horas, 'classification' para estrés binario
            target_column: columna objetivo a predecir
            representation: 'waveform' o 'spectrogram'
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_type = target_type
        self.target_column = target_column
        self.duration = duration
        self.sr = sr
        self.augment = augment
        self.representation = representation
        self.expected_length = int(duration * sr)
        
        if representation == 'spectrogram':
            self.n_mels = 128
        
        # Preparar encoder para targets categóricos
        if target_type == 'classification' and target_column in ['water_status']:
            self.label_encoder = LabelEncoder()
            self.dataframe[f'{target_column}_encoded'] = self.label_encoder.fit_transform(
                self.dataframe[target_column]
            )
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['audio_path']
        
        # Cargar target
        if self.target_type == 'regression':
            target = float(row[self.target_column])
        else:  # classification
            if f'{self.target_column}_encoded' in row:
                target = int(row[f'{self.target_column}_encoded'])
            else:
                target = int(row[self.target_column])
        
        try:
            # Cargar audio
            y, _ = librosa.load(audio_path, duration=self.duration, sr=self.sr)
            
            if len(y) == 0:
                if self.representation == 'waveform':
                    return torch.zeros(1, self.expected_length), torch.tensor(target, dtype=torch.float32 if self.target_type == 'regression' else torch.long)
                else:
                    expected_frames = int(self.duration * self.sr / 512) + 1
                    return torch.zeros(1, self.n_mels, expected_frames), torch.tensor(target, dtype=torch.float32 if self.target_type == 'regression' else torch.long)
            
            # Procesar según representación
            if self.representation == 'waveform':
                # Padding/truncating
                if len(y) < self.expected_length:
                    y = np.pad(y, (0, self.expected_length - len(y)), mode='constant')
                else:
                    y = y[:self.expected_length]
                
                # Normalización
                y = y / (np.max(np.abs(y)) + 1e-8)
                
                # Data augmentation para waveform
                if self.augment:
                    y = self._augment_waveform(y)
                
                audio_tensor = torch.FloatTensor(y).unsqueeze(0)
            
            else:  # spectrogram
                # Asegurar duración mínima
                min_length = int(self.duration * self.sr)
                if len(y) < min_length:
                    y = np.pad(y, (0, min_length - len(y)), mode='constant')
                else:
                    y = y[:min_length]
                
                # Crear espectrograma Mel
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=self.sr, n_mels=self.n_mels, 
                    n_fft=2048, hop_length=512
                )
                mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
                
                # Normalizar
                mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                
                # Data augmentation para espectrograma
                if self.augment:
                    mel_spec_db = self._augment_spectrogram(mel_spec_db)
                
                audio_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            
            target_tensor = torch.tensor(target, dtype=torch.float32 if self.target_type == 'regression' else torch.long)
            
            return audio_tensor, target_tensor
            
        except Exception as e:
            print(f"Error cargando {audio_path}: {e}")
            if self.representation == 'waveform':
                return torch.zeros(1, self.expected_length), torch.tensor(0.0 if self.target_type == 'regression' else 0, dtype=torch.float32 if self.target_type == 'regression' else torch.long)
            else:
                expected_frames = int(self.duration * self.sr / 512) + 1
                return torch.zeros(1, self.n_mels, expected_frames), torch.tensor(0.0 if self.target_type == 'regression' else 0, dtype=torch.float32 if self.target_type == 'regression' else torch.long)
    
    def _augment_waveform(self, y):
        """Augmentation específico para análisis de plantas"""
        if np.random.random() < 0.3:
            # Time shifting suave (las plantas pueden emitir en diferentes momentos)
            shift_samples = int(0.05 * len(y))
            shift = np.random.randint(-shift_samples, shift_samples)
            y = np.roll(y, shift)
        
        if np.random.random() < 0.2:
            # Volume augmentation muy suave (preservar características de intensidad)
            volume_factor = np.random.uniform(0.9, 1.1)
            y = y * volume_factor
        
        return y
    
    def _augment_spectrogram(self, spec):
        """Augmentation específico para espectrogramas de plantas"""
        if np.random.random() < 0.2:
            # Frequency masking muy suave
            freq_mask_param = max(1, int(0.05 * spec.shape[0]))
            freq_mask = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, spec.shape[0] - freq_mask))
            spec[f0:f0+freq_mask, :] = spec.mean()
        
        return spec

class PlantStressCNN(nn.Module):
    """CNN especializada para análisis de estrés hídrico en plantas"""
    
    def __init__(self, input_type='spectrogram', task_type='regression', num_classes=4, dropout=0.3):
        super(PlantStressCNN, self).__init__()
        
        self.input_type = input_type
        self.task_type = task_type
        
        if input_type == 'waveform':
            # CNN 1D para formas de onda
            self.features = nn.Sequential(
                # Bloque 1: Detección de eventos ultrasónicos
                nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=2),
                
                # Bloque 2: Patrones temporales
                nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=2),
                
                # Bloque 3: Características de estrés
                nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=2),
                
                # Pooling adaptativo
                nn.AdaptiveAvgPool1d(1)
            )
            feature_size = 128
            
        else:  # spectrogram
            # CNN 2D para espectrogramas
            self.features = nn.Sequential(
                # Bloque 1: Detección de eventos tiempo-frecuencia
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Bloque 2: Patrones espectrales
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Bloque 3: Características complejas
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Bloque 4: Especialización para estrés
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            feature_size = 128 * 4 * 4
        
        # Clasificador común
        if task_type == 'regression':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Salida escalar para regresión
            )
        else:  # classification
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        if self.task_type == 'regression':
            return x.squeeze()  # Remove last dimension for regression
        else:
            return x

class PlantStressGRU(nn.Module):
    """GRU mejorado para modelado temporal de estrés hídrico"""
    
    def __init__(self, task_type='regression', num_classes=4, hidden_size=128, num_layers=3, dropout=0.3):
        super(PlantStressGRU, self).__init__()
        
        self.task_type = task_type
        
        # Extractor de características CNN mejorado
        self.cnn_features = nn.Sequential(
            # Bloque 1: Captura de patrones de alta frecuencia
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Bloque 2: Patrones de media frecuencia
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Bloque 3: Patrones de baja frecuencia
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)  # Normalizar longitud
        )
        
        # GRU bidireccional con múltiples capas
        self.gru = nn.GRU(
            256, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Self-attention para identificar patrones críticos
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization layers
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )
        
        # Final classifier/regressor
        if task_type == 'regression':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, 512),  # hidden_size*2 + hidden_size*2 (avg+max pool)
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, 512),  # hidden_size*2 + hidden_size*2 (avg+max pool)
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        # Extracción de características CNN
        x = self.cnn_features(x)  # [batch, 256, 64]
        x = x.transpose(1, 2)     # [batch, 64, 256] para GRU
        
        # Procesamiento GRU
        gru_out, _ = self.gru(x)  # [batch, 64, hidden_size*2]
        
        # Self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        x = self.layer_norm1(gru_out + attn_out)  # Residual connection
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)  # Residual connection
        
        # Global pooling (average + max)
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Clasificación final
        x = self.classifier(x)
        
        if self.task_type == 'regression':
            return x.squeeze()
        else:
            return x

class PlantStressLSTM(nn.Module):
    """LSTM para modelado temporal de estrés hídrico"""
    
    def __init__(self, task_type='regression', num_classes=4, hidden_size=128, num_layers=2, dropout=0.3):
        super(PlantStressLSTM, self).__init__()
        
        self.task_type = task_type
        
        # Extractor de características CNN
        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # LSTM para modelado temporal
        self.lstm = nn.LSTM(
            128, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention para identificar momentos críticos
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Clasificador final
        if task_type == 'regression':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        # Extracción de características
        features = self.cnn_features(x)  # (batch, channels, time)
        features = features.permute(0, 2, 1)  # (batch, time, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Clasificación
        output = self.classifier(attended)
        
        if self.task_type == 'regression':
            return output.squeeze()
        else:
            return output

class PlantStressTransformer(nn.Module):
    """Transformer especializado para análisis de ultrasonidos de plantas"""
    
    def __init__(self, task_type='regression', num_classes=4, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(PlantStressTransformer, self).__init__()
        
        self.task_type = task_type
        self.d_model = d_model
        
        # Embedding layer para convertir audio en tokens
        self.audio_embedding = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head
        if task_type == 'regression':
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, num_classes)
            )
    
    def forward(self, x):
        # Embedding de audio
        x = self.audio_embedding(x)  # [batch, d_model, seq_len]
        x = x.transpose(1, 2)        # [batch, seq_len, d_model]
        
        # Positional encoding
        seq_len = x.size(1)
        if seq_len <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer processing
        x = self.layer_norm(x)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Classification
        x = self.head(x)
        
        if self.task_type == 'regression':
            return x.squeeze()
        else:
            return x

class PlantStressEnsemble(nn.Module):
    """Ensemble SEGURO sin RNN para evitar errores CUDA de memoria"""
    
    def __init__(self, task_type='regression', num_classes=4):
        super(PlantStressEnsemble, self).__init__()
        
        self.task_type = task_type
        
        # SOLO CNN y Transformer - SIN RNN para evitar crashes CUDA
        self.cnn_wave = PlantStressCNN(input_type='waveform', task_type=task_type, num_classes=num_classes)
        self.cnn_spec = PlantStressCNN(input_type='spectrogram', task_type=task_type, num_classes=num_classes)
        self.transformer = PlantStressTransformer(task_type=task_type, num_classes=num_classes, d_model=128, num_layers=2)
        
        # Meta-learner para combinar predicciones (3 modelos SIN RNN)
        input_size = 3 if task_type == 'regression' else 3 * num_classes
        if task_type == 'regression':
            self.meta_learner = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        else:
            self.meta_learner = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes)
            )
    
    def forward(self, x):
        # Obtener predicciones SOLO de modelos seguros (SIN RNN)
        pred_cnn_wave = self.cnn_wave(x)
        
        # Para CNN spectrogram, necesitamos convertir waveform a spectrogram
        if x.dim() == 3 and x.size(1) == 1:  # Waveform input [batch, 1, time]
            # Convertir a spectrogram para el segundo CNN
            import librosa
            x_spec_list = []
            for i in range(x.size(0)):
                # Convertir tensor a numpy
                waveform = x[i, 0, :].detach().cpu().numpy()
                # Crear spectrogram
                mel_spec = librosa.feature.melspectrogram(y=waveform, sr=22050, n_mels=128)
                mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
                # Normalizar
                mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                x_spec_list.append(mel_spec_db)
            
            # Convertir de vuelta a tensor
            x_spec = torch.FloatTensor(np.stack(x_spec_list)).unsqueeze(1).to(x.device)
            pred_cnn_spec = self.cnn_spec(x_spec)
        else:
            # Si ya es spectrogram, usar directamente
            pred_cnn_spec = self.cnn_spec(x)
        
        pred_transformer = self.transformer(x if x.dim() == 3 and x.size(1) == 1 else x)
        
        # Combinar predicciones de modelos SEGUROS
        if self.task_type == 'regression':
            # Para regresión, asegurar dimensiones correctas
            if pred_cnn_wave.dim() == 0:
                pred_cnn_wave = pred_cnn_wave.unsqueeze(0)
            if pred_cnn_wave.dim() == 1:
                pred_cnn_wave = pred_cnn_wave.unsqueeze(1)
                
            if pred_cnn_spec.dim() == 0:
                pred_cnn_spec = pred_cnn_spec.unsqueeze(0)
            if pred_cnn_spec.dim() == 1:
                pred_cnn_spec = pred_cnn_spec.unsqueeze(1)
                
            if pred_transformer.dim() == 0:
                pred_transformer = pred_transformer.unsqueeze(0)
            if pred_transformer.dim() == 1:
                pred_transformer = pred_transformer.unsqueeze(1)
            
            # Concatenar predicciones seguras
            combined = torch.cat([pred_cnn_wave, pred_cnn_spec, pred_transformer], dim=1)
        else:
            # Para clasificación, concatenar probabilidades
            combined = torch.cat([pred_cnn_wave, pred_cnn_spec, pred_transformer], dim=1)
        
        # Meta-learning
        output = self.meta_learner(combined)
        
        if self.task_type == 'regression':
            return output.squeeze()
        else:
            return output

class PlantStressWaveNet(nn.Module):
    """WaveNet adaptado para análisis de ultrasonidos de plantas (optimizado para memoria y dimensiones)"""
    
    def __init__(self, task_type='regression', num_classes=4, residual_channels=32, dilation_cycles=3, layers_per_cycle=6):
        super(PlantStressWaveNet, self).__init__()
        
        self.task_type = task_type
        self.residual_channels = residual_channels
        
        # Input projection
        self.start_conv = nn.Conv1d(1, residual_channels, kernel_size=1)
        
        # Dilated convolution blocks simplificados
        self.dilated_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for cycle in range(dilation_cycles):
            for layer in range(layers_per_cycle):
                dilation = 2 ** layer
                
                # Dilated convolution con padding correcto para mantener dimensiones
                self.dilated_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, 
                             kernel_size=3, dilation=dilation, padding=dilation)
                )
                
                # Gate convolution separada con mismas dimensiones
                self.gate_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, 
                             kernel_size=3, dilation=dilation, padding=dilation)
                )
                
                # Residual connection
                self.residual_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                )
                
                # Skip connection
                self.skip_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                )
        
        # Output layers
        self.output_conv1 = nn.Conv1d(residual_channels, residual_channels // 2, kernel_size=1)
        self.output_conv2 = nn.Conv1d(residual_channels // 2, residual_channels // 4, kernel_size=1)
        
        # Final classifier
        if task_type == 'regression':
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(residual_channels // 4, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(residual_channels // 4, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        # Input projection
        x = self.start_conv(x)
        
        # Skip connections accumulator
        skip_connections = []
        
        # Dilated convolution blocks con dimensiones consistentes
        for dilated_conv, gate_conv, residual_conv, skip_conv in zip(
            self.dilated_convs, self.gate_convs, self.residual_convs, self.skip_convs
        ):
            # Guardar input para residual connection
            residual_input = x
            
            # Dilated convolution
            conv_out = dilated_conv(x)
            gate_out = gate_conv(x)
            
            # Verificar que las dimensiones coincidan
            if conv_out.size() != gate_out.size():
                # Ajustar al mínimo tamaño si hay diferencias
                min_size = min(conv_out.size(-1), gate_out.size(-1))
                conv_out = conv_out[..., :min_size]
                gate_out = gate_out[..., :min_size]
            
            # Gating
            gated = torch.tanh(conv_out) * torch.sigmoid(gate_out)
            
            # Residual connection con verificación de dimensiones
            residual = residual_conv(gated)
            if residual.size(-1) != residual_input.size(-1):
                # Ajustar dimensiones para la conexión residual
                min_size = min(residual.size(-1), residual_input.size(-1))
                residual = residual[..., :min_size]
                residual_input = residual_input[..., :min_size]
            
            x = residual_input + residual
            
            # Skip connection
            skip = skip_conv(gated)
            skip_connections.append(skip)
        
        # Combine skip connections asegurando dimensiones consistentes
        if len(skip_connections) > 1:
            # Encontrar la dimensión mínima
            min_size = min(skip.size(-1) for skip in skip_connections)
            skip_connections = [skip[..., :min_size] for skip in skip_connections]
            
            # Sumar skip connections
            skip_sum = skip_connections[0]
            for skip in skip_connections[1:]:
                skip_sum = skip_sum + skip
        else:
            skip_sum = skip_connections[0]
        
        # Output processing
        x = torch.relu(skip_sum)
        x = torch.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        
        # Classification
        x = self.classifier(x)
        
        if self.task_type == 'regression':
            return x.squeeze()
        else:
            return x

# === FUNCIONES DE ENTRENAMIENTO ===

def train_plant_stress_model(model, train_loader, val_loader, task_type='regression', 
                           num_epochs=50, learning_rate=1e-3, model_name="PlantStress"):
    """Entrenar modelo para análisis de estrés hídrico"""
    
    # Decidir si usar DataParallel (algunos modelos son problemáticos)
    # ENSEMBLE DESHABILITADO para evitar crashes CUDA con RNN
    use_data_parallel = (use_multi_gpu and 
                        torch.cuda.device_count() > 1 and 
                        model_name.lower() not in ['ensemble_full', 'complex_ensemble', 'ensemble', 'lstm', 'gru'])
    
    if use_data_parallel:
        print(f"🎮 Usando DataParallel con {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    elif use_multi_gpu:
        print(f"🎮 Multi-GPU disponible pero usando GPU única para {model_name} (compatibilidad)")
    
    model = model.to(device)
    
    # Función de pérdida
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"🚀 Entrenando {model_name} para {task_type}")
    print(f"   Épocas: {num_epochs}, LR: {learning_rate}")
    print(f"   Dispositivo: {device}")
    
    # Configurar gradient accumulation para modelos pesados
    accumulation_steps = 1
    if 'wavenet' in model_name.lower() or 'ensemble' in model_name.lower():
        # Reducir accumulation si usamos DataParallel
        accumulation_steps = 2 if use_data_parallel else 4
        print(f"   🔧 Usando gradient accumulation con {accumulation_steps} pasos")
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_predictions, train_targets = [], []
        
        # Limpiar cache de GPU al inicio de cada época
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        optimizer.zero_grad()  # Mover fuera del loop para accumulation
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}")):
            try:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                
                if task_type == 'regression':
                    # Asegurar que output tiene dimensiones correctas para extend()
                    output_np = output.detach().cpu().numpy()
                    if output_np.ndim == 0:  # Tensor 0-dimensional
                        output_np = output_np.reshape(1)
                    train_predictions.extend(output_np)
                    
                    target_np = target.detach().cpu().numpy()
                    if target_np.ndim == 0:
                        target_np = target_np.reshape(1)
                    train_targets.extend(target_np)
                else:
                    _, predicted = torch.max(output, 1)
                    train_predictions.extend(predicted.detach().cpu().numpy())
                    train_targets.extend(target.detach().cpu().numpy())
                
                # Limpiar cache periódicamente
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  OOM en batch {batch_idx}, saltando...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    print(f"❌ Error en entrenamiento: {e}")
                    print(f"   Batch shape: {data.shape if 'data' in locals() else 'N/A'}")
                    print(f"   Modelo: {model_name}")
                    if 'ensemble' in model_name.lower():
                        print("   💡 Sugerencia: El ensemble puede tener problemas de compatibilidad")
                    raise e
        
        # Final gradient update si queda algo
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_predictions, val_targets = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    if task_type == 'regression':
                        # Asegurar que output tiene dimensiones correctas para extend()
                        output_np = output.cpu().numpy()
                        if output_np.ndim == 0:  # Tensor 0-dimensional
                            output_np = output_np.reshape(1)
                        val_predictions.extend(output_np)
                        
                        target_np = target.cpu().numpy()
                        if target_np.ndim == 0:
                            target_np = target_np.reshape(1)
                        val_targets.extend(target_np)
                    else:
                        _, predicted = torch.max(output, 1)
                        val_predictions.extend(predicted.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️  OOM en validación, saltando batch...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calcular métricas
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if task_type == 'regression':
            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            
            train_metric = train_r2
            val_metric = val_r2
            
            print(f"Época {epoch+1}: Train R²={train_r2:.3f} RMSE={train_rmse:.2f}, Val R²={val_r2:.3f} RMSE={val_rmse:.2f}")
        
        else:  # classification
            train_acc = np.mean(np.array(train_predictions) == np.array(train_targets))
            val_acc = np.mean(np.array(val_predictions) == np.array(val_targets))
            
            train_metric = train_acc
            val_metric = val_acc
            
            print(f"Época {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Guardar métricas
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Manejar DataParallel correctamente
            if hasattr(model, 'module'):
                best_model_state = model.module.state_dict().copy()
            else:
                best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"   🎯 Nuevo mejor modelo!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 15:
            print(f"Early stopping en época {epoch+1}")
            break
        
        # Limpiar cache al final de cada época
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Cargar mejor modelo
    if best_model_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss
    }

def evaluate_plant_stress_model(model, test_loader, task_type='regression', model_name="Model"):
    """Evaluar modelo de estrés hídrico"""
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluando {model_name}"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if task_type == 'regression':
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
            else:
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.cpu().numpy())
                targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    if task_type == 'regression':
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = np.mean(np.abs(targets - predictions))
        
        print(f"\n📊 {model_name} - Regresión:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    else:  # classification
        accuracy = np.mean(predictions == targets)
        
        print(f"\n📊 {model_name} - Clasificación:")
        print(f"   Accuracy: {accuracy:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(targets, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'targets': targets,
            'classification_report': classification_report(targets, predictions, output_dict=True)
        }

# === ANÁLISIS EXPLORATORIO Y VISUALIZACIONES ===

def analyze_plant_ultrasound_patterns(df, save_path=None):
    """Análisis exploratorio de patrones en ultrasonidos de plantas"""
    
    print("\n🔬 ANÁLISIS DE PATRONES EN ULTRASONIDOS DE PLANTAS")
    print("="*60)
    
    # Configurar matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis de Ultrasonidos de Plantas vs Estado Hídrico', fontsize=16, fontweight='bold')
    
    # 1. Distribución temporal de emisiones
    ax1 = axes[0, 0]
    df['hour'] = df['recording_time'].dt.hour
    hourly_counts = df.groupby('hour').size()
    ax1.bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='green')
    ax1.set_title('Emisiones Ultrasónicas por Hora del Día')
    ax1.set_xlabel('Hora del día')
    ax1.set_ylabel('Número de emisiones')
    ax1.grid(True, alpha=0.3)
    
    # 2. Tiempo desde último riego vs número de emisiones
    ax2 = axes[0, 1]
    bins = np.arange(0, df['hours_since_watering'].max() + 12, 12)
    df['time_bin'] = pd.cut(df['hours_since_watering'], bins=bins, labels=[f"{int(bins[i])}-{int(bins[i+1])}h" for i in range(len(bins)-1)])
    time_counts = df.groupby('time_bin').size()
    ax2.bar(range(len(time_counts)), time_counts.values, alpha=0.7, color='orange')
    ax2.set_title('Emisiones vs Tiempo desde Último Riego')
    ax2.set_xlabel('Tiempo desde último riego')
    ax2.set_ylabel('Número de emisiones')
    ax2.set_xticks(range(len(time_counts)))
    ax2.set_xticklabels(time_counts.index, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución por especie
    ax3 = axes[0, 2]
    species_counts = df['species'].value_counts()
    ax3.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Distribución por Especies')
    
    # 4. Patrón temporal detallado (horas vs días)
    ax4 = axes[1, 0]
    scatter = ax4.scatter(df['days_since_watering'], df['hour'], 
                         c=df['hours_since_watering'], cmap='RdYlBu_r', alpha=0.6)
    ax4.set_title('Patrón Temporal: Día vs Hora de Emisión')
    ax4.set_xlabel('Días desde último riego')
    ax4.set_ylabel('Hora del día')
    plt.colorbar(scatter, ax=ax4, label='Horas desde riego')
    
    # 5. Distribución de estrés hídrico
    ax5 = axes[1, 1]
    if 'water_status' in df.columns:
        status_counts = df['water_status'].value_counts()
        colors = ['lightgreen', 'yellow', 'orange', 'red']
        ax5.bar(status_counts.index, status_counts.values, 
                color=colors[:len(status_counts)], alpha=0.7)
        ax5.set_title('Distribución por Estado Hídrico')
        ax5.set_ylabel('Número de emisiones')
        ax5.tick_params(axis='x', rotation=45)
    
    # 6. Emisiones por planta individual
    ax6 = axes[1, 2]
    plant_counts = df.groupby('plant_id').size().sort_values(ascending=False).head(10)
    ax6.bar(range(len(plant_counts)), plant_counts.values, alpha=0.7, color='purple')
    ax6.set_title('Top 10 Plantas con Más Emisiones')
    ax6.set_xlabel('Planta ID')
    ax6.set_ylabel('Número de emisiones')
    ax6.set_xticks(range(len(plant_counts)))
    ax6.set_xticklabels(plant_counts.index, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'plant_ultrasound_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Gráficos guardados en: {save_path / 'plant_ultrasound_analysis.png'}")
    
    plt.show()
    
    # Estadísticas adicionales
    print(f"\n📊 ESTADÍSTICAS DETALLADAS:")
    print(f"   Emisiones totales: {len(df)}")
    print(f"   Rango temporal: {df['recording_time'].min()} a {df['recording_time'].max()}")
    print(f"   Tiempo máximo sin riego: {df['hours_since_watering'].max():.1f} horas ({df['days_since_watering'].max():.1f} días)")
    print(f"   Promedio horas sin riego: {df['hours_since_watering'].mean():.1f} ± {df['hours_since_watering'].std():.1f}")
    
    # Correlaciones interesantes
    if len(df) > 1:
        print(f"\n🔍 CORRELACIONES:")
        print(f"   Emisiones vs Horas sin riego: {df.groupby(pd.cut(df['hours_since_watering'], bins=5)).size().corr(pd.Series(range(5))):.3f}")
        print(f"   Emisiones vs Hora del día: {df.groupby('hour').size().corr(pd.Series(range(24))):.3f}")

def create_water_stress_timeline(df, save_path=None):
    """Crear timeline de estrés hídrico por planta"""
    
    plt.figure(figsize=(15, 10))
    
    unique_plants = df['plant_id'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_plants)))
    
    for i, plant in enumerate(unique_plants):
        plant_data = df[df['plant_id'] == plant].sort_values('recording_time')
        
        # Timeline de emisiones
        y_pos = [i] * len(plant_data)
        plt.scatter(plant_data['recording_time'], y_pos, 
                   c=plant_data['hours_since_watering'], 
                   cmap='RdYlBu_r', s=50, alpha=0.7, 
                   label=f'Planta {plant}' if i < 10 else "")
    
    plt.colorbar(label='Horas desde último riego')
    plt.title('Timeline de Emisiones Ultrasónicas por Planta')
    plt.xlabel('Tiempo de grabación')
    plt.ylabel('Planta')
    plt.yticks(range(len(unique_plants)), [f'Planta {p}' for p in unique_plants])
    
    if len(unique_plants) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'water_stress_timeline.png', dpi=300, bbox_inches='tight')
        print(f"💾 Timeline guardado en: {save_path / 'water_stress_timeline.png'}")
    
    plt.show()

# === VISUALIZACIÓN DE RESULTADOS CONJUNTOS ===

def visualize_model_comparison(results_dir, task_type='regression'):
    """Visualizar comparación de todos los modelos entrenados"""
    
    results_dir = Path(results_dir)
    results_file = results_dir / 'results_summary.json'
    
    if not results_file.exists():
        print(f"❌ No se encontró archivo de resultados: {results_file}")
        print("   Ejecuta primero el entrenamiento de modelos.")
        return
    
    # Cargar resultados
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("❌ No hay resultados para visualizar")
        return
    
    print(f"\n📊 VISUALIZACIÓN COMPARATIVA DE MODELOS - {task_type.upper()}")
    print("="*60)
    
    model_names = list(results.keys())
    
    # Función auxiliar para convertir valores a float de forma segura
    def safe_float(value):
        """Convierte un valor a float de forma segura"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        else:
            return 0.0
    
    if task_type == 'regression':
        # Para regresión: R², RMSE, MAE - Convertir a float de forma segura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Modelos para Predicción de Estrés Hídrico', fontsize=16, fontweight='bold')
        
        # R² Score
        ax1 = axes[0, 0]
        r2_scores = [safe_float(results[model]['r2_score']) for model in model_names]
        bars1 = ax1.bar(model_names, r2_scores, alpha=0.7, color='skyblue')
        ax1.set_title('R² Score (Mayor = Mejor)')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE
        ax2 = axes[0, 1]
        rmse_scores = [safe_float(results[model]['rmse']) for model in model_names]
        bars2 = ax2.bar(model_names, rmse_scores, alpha=0.7, color='lightcoral')
        ax2.set_title('RMSE (Menor = Mejor)')
        ax2.set_ylabel('RMSE')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            max_rmse = max(rmse_scores) if rmse_scores else 1.0
            ax2.text(bar.get_x() + bar.get_width()/2., height + max_rmse*0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # MAE
        ax3 = axes[1, 0]
        mae_scores = [safe_float(results[model]['mae']) for model in model_names]
        bars3 = ax3.bar(model_names, mae_scores, alpha=0.7, color='lightgreen')
        ax3.set_title('MAE (Menor = Mejor)')
        ax3.set_ylabel('MAE')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, score in zip(bars3, mae_scores):
            height = bar.get_height()
            max_mae = max(mae_scores) if mae_scores else 1.0
            ax3.text(bar.get_x() + bar.get_width()/2., height + max_mae*0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Radar chart para comparación general
        ax4 = axes[1, 1]
        
        # Normalizar métricas para radar chart (invertir RMSE y MAE)
        r2_norm = np.array(r2_scores)
        max_rmse = max(rmse_scores) if rmse_scores else 1.0
        max_mae = max(mae_scores) if mae_scores else 1.0
        rmse_norm = 1 - (np.array(rmse_scores) / max_rmse)  # Invertir
        mae_norm = 1 - (np.array(mae_scores) / max_mae)    # Invertir
        
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, model in enumerate(model_names):
            values = [r2_norm[i], rmse_norm[i], mae_norm[i]]
            values += values[:1]  # Cerrar el círculo
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax4.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(['R²', 'RMSE (inv)', 'MAE (inv)'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Comparación General\n(Valores más altos = mejor)')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Ranking de modelos
        print(f"\n🏆 RANKING DE MODELOS (Regresión):")
        print("-" * 40)
        
        # Calcular score combinado (ponderado)
        combined_scores = []
        for i, model in enumerate(model_names):
            # R² tiene peso 50%, RMSE 30%, MAE 20%
            score = (r2_scores[i] * 0.5 + 
                    rmse_norm[i] * 0.3 + 
                    mae_norm[i] * 0.2)
            combined_scores.append((model, score, r2_scores[i], rmse_scores[i], mae_scores[i]))
        
        # Ordenar por score combinado
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, score, r2, rmse, mae) in enumerate(combined_scores, 1):
            print(f"{rank}. {model}")
            print(f"   Score combinado: {score:.3f}")
            print(f"   R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            print()
    
    else:  # classification
        # Para clasificación: Accuracy - Convertir a float de forma segura
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Comparación de Modelos para Clasificación de Estrés Hídrico', fontsize=16, fontweight='bold')
        
        # Accuracy
        ax1 = axes[0]
        accuracies = [safe_float(results[model]['accuracy']) for model in model_names]
        bars = ax1.bar(model_names, accuracies, alpha=0.7, color='gold')
        ax1.set_title('Accuracy (Mayor = Mejor)')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Gráfico de pizza con porcentajes
        ax2 = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        wedges, texts, autotexts = ax2.pie(accuracies, labels=model_names, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Distribución de Accuracy')
        
        # Ranking para clasificación
        print(f"\n🏆 RANKING DE MODELOS (Clasificación):")
        print("-" * 40)
        
        model_acc_pairs = list(zip(model_names, accuracies))
        model_acc_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, acc) in enumerate(model_acc_pairs, 1):
            print(f"{rank}. {model}: {acc:.3f}")
    
    plt.tight_layout()
    
    # Guardar visualización
    save_path = results_dir / f'model_comparison_{task_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualización guardada en: {save_path}")
    
    plt.show()
    
    # Mostrar estadísticas adicionales
    print(f"\n📈 ESTADÍSTICAS ADICIONALES:")
    print("-" * 30)
    print(f"   Modelos evaluados: {len(model_names)}")
    print(f"   Directorio de resultados: {results_dir}")
    
    if task_type == 'regression':
        best_r2_model = max(model_names, key=lambda x: safe_float(results[x]['r2_score']))
        best_rmse_model = min(model_names, key=lambda x: safe_float(results[x]['rmse']))
        print(f"   Mejor R²: {best_r2_model} ({safe_float(results[best_r2_model]['r2_score']):.3f})")
        print(f"   Mejor RMSE: {best_rmse_model} ({safe_float(results[best_rmse_model]['rmse']):.3f})")
    else:
        best_acc_model = max(model_names, key=lambda x: safe_float(results[x]['accuracy']))
        print(f"   Mejor Accuracy: {best_acc_model} ({safe_float(results[best_acc_model]['accuracy']):.3f})")

# === FUNCIÓN PRINCIPAL ===

def main():
    """Función principal para análisis de ultrasonidos de plantas"""
    
    parser = argparse.ArgumentParser(description="Análisis de Ultrasonidos de Plantas para Predicción de Estado Hídrico")
    
    # Configuración de datos
    parser.add_argument('--data-path', type=str, default='data',
                       help='Ruta al directorio de datos')
    parser.add_argument('--stress-threshold', type=float, default=72.0,
                       help='Umbral en horas para considerar estrés hídrico')
    
    # Configuración de modelos
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn_waveform', 'cnn_spectrogram', 'lstm', 'gru', 'transformer', 'wavenet', 'ensemble', 'all'],
                       default=['all'],
                       help='Modelos a entrenar')
    parser.add_argument('--task-type', type=str, choices=['regression', 'classification'], 
                       default='regression',
                       help='Tipo de tarea: regresión (horas) o clasificación (estrés)')
    
    # Configuración de entrenamiento
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Tamaño del batch (se ajusta automáticamente si hay problemas de memoria)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Análisis exploratorio
    parser.add_argument('--analyze-only', action='store_true',
                       help='Solo realizar análisis exploratorio')
    
    # Visualización de resultados
    parser.add_argument('--visualize-results', action='store_true',
                       help='Visualizar resultados comparativos de todos los modelos entrenados')
    
    # Directorios de salida
    parser.add_argument('--results-dir', type=str, default='../results_task2',
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    print("🌱 ANÁLISIS DE ULTRASONIDOS DE PLANTAS - TAREA 2")
    print("="*60)
    print(f"Objetivo: Predecir estrés hídrico en plantas mediante ultrasonidos")
    print(f"Tipo de tarea: {args.task_type}")
    print(f"Umbral de estrés: {args.stress_threshold} horas")
    
    # Crear directorio de resultados
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    data_path = Path(args.data_path)
    metadata, classifications = load_metadata_and_classification(data_path)
    
    if metadata.empty or classifications.empty:
        print("❌ No se pudieron cargar los datos")
        return
    
    # Crear dataset de ultrasonidos de plantas
    plant_df = create_plant_ultrasound_dataset(data_path, metadata, classifications)
    
    if plant_df.empty:
        print("❌ No se encontraron ultrasonidos de plantas válidos")
        return
    
    # Crear etiquetas de estrés hídrico
    plant_df = create_water_stress_labels(plant_df, args.stress_threshold)
    
    # Análisis exploratorio
    print("\n🔬 Realizando análisis exploratorio...")
    analyze_plant_ultrasound_patterns(plant_df, results_dir)
    create_water_stress_timeline(plant_df, results_dir)
    
    # Guardar dataset procesado
    dataset_path = results_dir / 'plant_ultrasound_dataset.csv'
    plant_df.to_csv(dataset_path, index=False)
    print(f"💾 Dataset guardado en: {dataset_path}")
    
    if args.analyze_only:
        print("✅ Análisis exploratorio completado")
        return
    
    # Comando para visualizar resultados
    if args.visualize_results:
        print("📊 Visualizando resultados comparativos de modelos...")
        visualize_model_comparison(results_dir, args.task_type)
        return
    
    # Preparar datos para entrenamiento
    if len(plant_df) < 50:
        print("⚠️  Dataset muy pequeño para entrenamiento de Deep Learning")
        print("   Recomendado: usar análisis estadístico o algoritmos tradicionales")
        return
    
    # Determinar target y configuración
    if args.task_type == 'regression':
        target_column = 'hours_since_watering'
        num_classes = 1
    else:
        target_column = 'water_stress'  # o 'water_status' para multi-clase
        num_classes = len(plant_df[target_column].unique())
    
    # División de datos
    train_df, temp_df = train_test_split(plant_df, test_size=0.3, random_state=42, stratify=None)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=None)
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {len(train_df)} muestras")
    print(f"   Validación: {len(val_df)} muestras")
    print(f"   Prueba: {len(test_df)} muestras")
    
    # Definir modelos a entrenar
    available_models = {
        'cnn_waveform': ('waveform', PlantStressCNN),
        'cnn_spectrogram': ('spectrogram', PlantStressCNN),
        'lstm': ('waveform', PlantStressLSTM),
        'gru': ('waveform', PlantStressGRU),
        'transformer': ('waveform', PlantStressTransformer),
        'wavenet': ('waveform', PlantStressWaveNet),
        'ensemble': ('waveform', PlantStressEnsemble)
    }
    
    if 'all' in args.models:
        models_to_train = list(available_models.keys())
    else:
        models_to_train = args.models
    
    print(f"\n🎯 Modelos a entrenar: {models_to_train}")
    
    # Entrenar modelos
    all_results = {}
    
    for model_name in models_to_train:
        if model_name not in available_models:
            continue
            
        input_type, model_class = available_models[model_name]
        
        print(f"\n{'='*20} ENTRENANDO {model_name.upper()} {'='*20}")
        
        # Crear datasets
        train_dataset = PlantStressDataset(train_df, args.task_type, target_column, 
                                         representation=input_type, augment=True)
        val_dataset = PlantStressDataset(val_df, args.task_type, target_column, 
                                       representation=input_type, augment=False)
        test_dataset = PlantStressDataset(test_df, args.task_type, target_column, 
                                        representation=input_type, augment=False)
        
        # Ajustar batch size según el modelo para evitar OOM
        batch_size = args.batch_size
        if model_name in ['wavenet', 'ensemble']:
            batch_size = max(2, args.batch_size // 4)  # Reducir drasticamente para modelos pesados
            print(f"   🔧 Batch size ajustado a {batch_size} para {model_name}")
        elif model_name in ['transformer', 'gru']:
            batch_size = max(4, args.batch_size // 2)  # Reducir moderadamente
            print(f"   🔧 Batch size ajustado a {batch_size} para {model_name}")
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Crear modelo
        if model_class == PlantStressCNN:
            model = model_class(input_type=input_type, task_type=args.task_type, 
                              num_classes=num_classes)
        else:  # LSTM
            model = model_class(task_type=args.task_type, num_classes=num_classes)
        
        # Entrenar
        train_results = train_plant_stress_model(
            model, train_loader, val_loader, args.task_type,
            args.epochs, args.learning_rate, model_name
        )
        
        # Evaluar
        eval_results = evaluate_plant_stress_model(
            train_results['model'], test_loader, args.task_type, model_name
        )
        
        # Combinar resultados
        all_results[model_name] = {**train_results, **eval_results}
        
        # Guardar modelo
        model_path = results_dir / f"{model_name}_model.pt"
        torch.save(train_results['model'].state_dict(), model_path)
        print(f"💾 Modelo guardado: {model_path}")
    
    # Guardar resultados finales
    results_summary = {}
    for model_name, results in all_results.items():
        if args.task_type == 'regression':
            results_summary[model_name] = {
                'r2_score': results['r2_score'],
                'rmse': results['rmse'],
                'mae': results['mae']
            }
        else:
            results_summary[model_name] = {
                'accuracy': results['accuracy']
            }
    
    # Guardar resumen
    with open(results_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Mostrar resumen final
    print(f"\n🏆 RESUMEN FINAL - {args.task_type.upper()}")
    print("="*50)
    
    for model_name, metrics in results_summary.items():
        print(f"\n📊 {model_name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
    
    print(f"\n✅ Análisis completado. Resultados en: {results_dir}")

if __name__ == "__main__":
    main()
