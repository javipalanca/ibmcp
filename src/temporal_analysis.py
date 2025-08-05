"""
Análisis Temporal de Ultrasonidos de Plantas
============================================

Este script implementa análisis temporal específico para entender:
1. Patrones de evolución de sonidos durante la deshidratación
2. Predicción de días sin agua basada en características acústicas
3. Clustering de patrones sonoros por nivel de estrés
4. Análisis de correlaciones temporales

Autor: AI Assistant
Fecha: Agosto 2025
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import librosa
import librosa.display

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import PlantAudioDataset
from src.models.deep_learning_models import SpectrogramCNN
from src.models.traditional_ml_models import AdvancedFeatureExtractor
from src.evaluation.metrics import ModelEvaluator
from src.utils.training_utils import EarlyStopping, set_random_seed

class TemporalDataset(Dataset):
    """Dataset para análisis temporal de secuencias de audio."""
    
    def __init__(self, data_path: str, sequence_length: int = 10, 
                 temporal_stride: int = 1, only_valid: bool = True):
        """
        Args:
            data_path: Ruta al dataset
            sequence_length: Longitud de las secuencias temporales
            temporal_stride: Paso entre secuencias
            only_valid: Si solo usar sonidos válidos (label=1)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.only_valid = only_valid
        
        # Cargar metadata
        self.metadata = self._load_metadata()
        self.sequences = self._create_temporal_sequences()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Cargar y procesar metadata del dataset."""
        # Aquí implementarías la carga específica de tu metadata
        # Por ahora, crear un ejemplo de estructura
        metadata_path = self.data_path / "metadata.tsv"
        if metadata_path.exists():
            df = pd.read_csv(metadata_path, sep='\t')
        else:
            # Crear metadata de ejemplo basada en nombres de archivos
            image_files = list((self.data_path / "spectrograms").glob("*.jpg"))
            df = pd.DataFrame({
                'filename': [f.name for f in image_files],
                'label': np.random.randint(0, 2, len(image_files)),
                'channel': np.random.randint(1, 5, len(image_files)),
                'days_without_water': np.random.randint(0, 15, len(image_files)),
                'timestamp': pd.date_range('2025-01-01', periods=len(image_files), freq='1min')
            })
        
        if self.only_valid:
            df = df[df['label'] == 1].copy()
            
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def _create_temporal_sequences(self) -> List[Dict]:
        """Crear secuencias temporales para análisis."""
        sequences = []
        
        # Agrupar por planta/canal
        for channel in self.metadata['channel'].unique():
            channel_data = self.metadata[self.metadata['channel'] == channel]
            
            # Crear secuencias deslizantes
            for i in range(0, len(channel_data) - self.sequence_length + 1, self.temporal_stride):
                sequence = channel_data.iloc[i:i + self.sequence_length]
                
                sequences.append({
                    'files': sequence['filename'].tolist(),
                    'labels': sequence['label'].tolist(),
                    'days_without_water': sequence['days_without_water'].tolist(),
                    'channel': channel,
                    'start_idx': i,
                    'avg_days': sequence['days_without_water'].mean(),
                    'stress_progression': sequence['days_without_water'].iloc[-1] - sequence['days_without_water'].iloc[0]
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Cargar espectrogramas de la secuencia
        spectrograms = []
        for filename in sequence['files']:
            img_path = self.data_path / "spectrograms" / filename
            if img_path.exists():
                img = plt.imread(str(img_path))
                spectrograms.append(img)
        
        # Convertir a tensor
        spectrograms = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in spectrograms])
        
        return {
            'spectrograms': spectrograms,
            'labels': torch.tensor(sequence['labels']),
            'days_without_water': torch.tensor(sequence['days_without_water'], dtype=torch.float32),
            'avg_days': torch.tensor(sequence['avg_days'], dtype=torch.float32),
            'stress_progression': torch.tensor(sequence['stress_progression'], dtype=torch.float32),
            'channel': sequence['channel']
        }

class TemporalLSTM(nn.Module):
    """LSTM para análisis temporal de características espectrales."""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128, 
                 num_layers: int = 3, num_classes: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        
        self.feature_extractor = SpectrogramCNN(model_name='resnet18', num_classes=input_size)
        self.feature_extractor.classifier = nn.Linear(self.feature_extractor.classifier.in_features, input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.shape[:2]
        
        # Extraer features para cada frame
        x = x.view(-1, *x.shape[2:])  # (batch_size * seq_len, channels, height, width)
        features = self.feature_extractor(x)  # (batch_size * seq_len, input_size)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, input_size)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Usar la salida del último paso temporal
        output = self.classifier(lstm_out[:, -1, :])
        
        return output

class DaysWithoutWaterPredictor(nn.Module):
    """Modelo para predecir días sin agua basado en espectrogramas."""
    
    def __init__(self, backbone: str = 'resnet50', num_features: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.feature_extractor = SpectrogramCNN(model_name=backbone, num_classes=num_features)
        self.feature_extractor.classifier = nn.Linear(self.feature_extractor.classifier.in_features, num_features)
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features // 2, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, 1),
            nn.ReLU()  # Para asegurar valores positivos
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        days = self.regressor(features)
        return days.squeeze()

class TemporalAnalyzer:
    """Analizador principal para análisis temporal."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Inicializar componentes
        self.feature_extractor = AdvancedFeatureExtractor()
        self.evaluator = ModelEvaluator()
        
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"temporal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def analyze_temporal_patterns(self, data_path: str) -> Dict:
        """Analizar patrones temporales en el dataset."""
        self.logger.info("Iniciando análisis de patrones temporales...")
        
        # Cargar dataset temporal
        dataset = TemporalDataset(
            data_path=data_path,
            sequence_length=self.config['data']['sequence_length'],
            temporal_stride=self.config['data']['temporal_stride'],
            only_valid=self.config['data']['only_valid_sounds']
        )
        
        results = {}
        
        # Análisis de clustering
        if self.config['analysis']['clustering']['enabled']:
            results['clustering'] = self._perform_clustering_analysis(dataset)
        
        # Análisis de correlaciones
        results['correlations'] = self._analyze_correlations(dataset)
        
        # Visualizaciones
        results['visualizations'] = self._create_temporal_visualizations(dataset)
        
        return results
    
    def _perform_clustering_analysis(self, dataset: TemporalDataset) -> Dict:
        """Realizar análisis de clustering en los patrones temporales."""
        self.logger.info("Realizando análisis de clustering...")
        
        # Extraer features para clustering
        all_features = []
        all_days = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            # Procesar primera imagen de la secuencia para features
            img = sample['spectrograms'][0].numpy().transpose(1, 2, 0)
            features = self.feature_extractor.extract_spectrogram_features(img)
            
            all_features.append(features)
            all_days.append(sample['avg_days'].item())
        
        X = np.array(all_features)
        days = np.array(all_days)
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clustering_results = {}
        
        # Probar diferentes algoritmos de clustering
        for algorithm in self.config['analysis']['clustering']['algorithms']:
            if algorithm == 'kmeans':
                for n_clusters in self.config['analysis']['clustering']['n_clusters']:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(X_scaled)
                    
                    # Analizar clusters
                    cluster_analysis = self._analyze_clusters(labels, days, n_clusters)
                    clustering_results[f'kmeans_{n_clusters}'] = cluster_analysis
            
            elif algorithm == 'dbscan':
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    cluster_analysis = self._analyze_clusters(labels, days, n_clusters)
                    clustering_results['dbscan'] = cluster_analysis
        
        return clustering_results
    
    def _analyze_clusters(self, labels: np.ndarray, days: np.ndarray, n_clusters: int) -> Dict:
        """Analizar la calidad y características de los clusters."""
        cluster_stats = {}
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                cluster_days = days[mask]
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': np.sum(mask),
                    'avg_days': np.mean(cluster_days),
                    'std_days': np.std(cluster_days),
                    'min_days': np.min(cluster_days),
                    'max_days': np.max(cluster_days)
                }
        
        return {
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'silhouette_score': self._calculate_silhouette_score(labels, days)
        }
    
    def _calculate_silhouette_score(self, labels: np.ndarray, days: np.ndarray) -> float:
        """Calcular silhouette score para evaluar calidad del clustering."""
        from sklearn.metrics import silhouette_score
        
        # Usar días como feature simple para silhouette score
        if len(set(labels)) > 1:
            return silhouette_score(days.reshape(-1, 1), labels)
        return 0.0
    
    def _analyze_correlations(self, dataset: TemporalDataset) -> Dict:
        """Analizar correlaciones entre features acústicas y días sin agua."""
        self.logger.info("Analizando correlaciones temporales...")
        
        correlations = {}
        # Implementar análisis de correlaciones específico
        
        return correlations
    
    def _create_temporal_visualizations(self, dataset: TemporalDataset) -> Dict:
        """Crear visualizaciones de patrones temporales."""
        self.logger.info("Creando visualizaciones temporales...")
        
        # Crear directorio para visualizaciones
        vis_dir = Path(self.config['experiment']['save_dir']) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        # Ejemplo: Evolución temporal de características
        self._plot_temporal_evolution(dataset, vis_dir)
        
        return {'save_dir': str(vis_dir)}
    
    def _plot_temporal_evolution(self, dataset: TemporalDataset, save_dir: Path):
        """Plotear evolución temporal de características."""
        # Implementar visualización específica
        pass
    
    def train_temporal_models(self, data_path: str) -> Dict:
        """Entrenar modelos específicos para análisis temporal."""
        self.logger.info("Entrenando modelos temporales...")
        
        # Cargar dataset
        dataset = TemporalDataset(data_path=data_path)
        
        # Dividir en train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'])
        
        results = {}
        
        # Entrenar modelos temporales
        for model_config in self.config['models']['temporal_models']:
            model_name = model_config['architecture']
            self.logger.info(f"Entrenando modelo: {model_name}")
            
            model = self._create_temporal_model(model_config)
            trainer_results = self._train_single_model(model, train_loader, val_loader, model_name)
            results[model_name] = trainer_results
        
        # Entrenar modelos de regresión
        for model_config in self.config['models']['regression_models']:
            model_name = model_config['architecture']
            self.logger.info(f"Entrenando modelo de regresión: {model_name}")
            
            model = self._create_regression_model(model_config)
            trainer_results = self._train_regression_model(model, train_loader, val_loader, model_name)
            results[model_name] = trainer_results
        
        return results
    
    def _create_temporal_model(self, config: Dict) -> nn.Module:
        """Crear modelo temporal basado en configuración."""
        if config['type'] == 'rnn':
            return TemporalLSTM(**config['params'])
        else:
            raise ValueError(f"Tipo de modelo temporal no soportado: {config['type']}")
    
    def _create_regression_model(self, config: Dict) -> nn.Module:
        """Crear modelo de regresión basado en configuración."""
        if config['task'] == 'predict_days_without_water':
            return DaysWithoutWaterPredictor(**config['params'])
        else:
            raise ValueError(f"Tarea de regresión no soportada: {config['task']}")
    
    def _train_single_model(self, model: nn.Module, train_loader: DataLoader, 
                           val_loader: DataLoader, model_name: str) -> Dict:
        """Entrenar un modelo individual."""
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=self.config['training']['early_stopping_patience'])
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                spectrograms = batch['spectrograms'].to(self.device)
                labels = batch['labels'][:, -1].to(self.device)  # Usar último label de la secuencia
                
                optimizer.zero_grad()
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    spectrograms = batch['spectrograms'].to(self.device)
                    labels = batch['labels'][:, -1].to(self.device)
                    
                    outputs = model(spectrograms)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping en epoch {epoch+1}")
                break
        
        # Guardar modelo
        save_path = Path(self.config['experiment']['save_dir']) / f"{model_name}_temporal.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': str(save_path),
            'best_val_loss': min(val_losses)
        }
    
    def _train_regression_model(self, model: nn.Module, train_loader: DataLoader, 
                               val_loader: DataLoader, model_name: str) -> Dict:
        """Entrenar modelo de regresión."""
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                # Usar primer espectrograma de la secuencia para regresión
                spectrograms = batch['spectrograms'][:, 0].to(self.device)
                targets = batch['avg_days'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(spectrograms)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    spectrograms = batch['spectrograms'][:, 0].to(self.device)
                    targets = batch['avg_days'].to(self.device)
                    
                    outputs = model(spectrograms)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Calcular métricas de regresión
            mae = mean_absolute_error(val_targets, val_predictions)
            r2 = r2_score(val_targets, val_predictions)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Guardar modelo
        save_path = Path(self.config['experiment']['save_dir']) / f"{model_name}_regression.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': str(save_path),
            'final_mae': mae,
            'final_r2': r2
        }

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Análisis Temporal de Ultrasonidos de Plantas')
    parser.add_argument('--config', type=str, default='configs/temporal_analysis_config.yaml',
                        help='Ruta al archivo de configuración')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Ruta al dataset')
    parser.add_argument('--mode', type=str, choices=['analyze', 'train', 'both'], default='both',
                        help='Modo de ejecución')
    
    args = parser.parse_args()
    
    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configurar semilla aleatoria
    set_random_seed(config['experiment']['random_seed'])
    
    # Crear analizador
    analyzer = TemporalAnalyzer(config)
    
    results = {}
    
    if args.mode in ['analyze', 'both']:
        # Realizar análisis temporal
        analysis_results = analyzer.analyze_temporal_patterns(args.data_path)
        results['analysis'] = analysis_results
    
    if args.mode in ['train', 'both']:
        # Entrenar modelos temporales
        training_results = analyzer.train_temporal_models(args.data_path)
        results['training'] = training_results
    
    # Guardar resultados
    results_path = Path(config['experiment']['save_dir']) / 'temporal_analysis_results.yaml'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"✅ Análisis temporal completado. Resultados guardados en: {results_path}")

if __name__ == "__main__":
    main()
