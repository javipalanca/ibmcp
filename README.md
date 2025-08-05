# Plant Ultrasonic Audio Classification (PUAC)

Un proyecto de machine learning avanzado para clasificar ultrasonidos de plantas y analizar patrones de estrés hídrico utilizando técnicas de deep learning y ensemble methods.

## Descripción del Proyecto

Este proyecto implementa múltiples modelos de machine learning para:

1. **Clasificación de Validez**: Distinguir entre sonidos válidos emitidos por plantas (etiqueta 1) vs ruido ambiental o inválido (etiqueta 0)
2. **Análisis de Patrones**: Entender los patrones temporales de los ultrasonidos durante el proceso de deshidratación
3. **Predicción de Estrés**: Potencialmente predecir cuántos días lleva una planta sin agua basándose en sus patrones sonoros

## Metodología de Captura

- **Setup experimental**: 4 plantas monitoreadas simultáneamente (una por canal/micrófono)
- **Protocolo**: Las plantas dejan de ser regadas al inicio del muestreo
- **Discriminación de ruido**: 
  - Sonidos capturados en múltiples micrófonos = ruido ambiental
  - Sonidos capturados en un solo micrófono = posible emisión de planta
- **Duración**: Seguimiento durante varios días de deshidratación progresiva

## Dataset

- **126,678 muestras** etiquetadas
- **Clasificación binaria**: Inválido/Ruido (0) vs Sonido válido de planta (1)
- **Espectrogramas** en formato JPG
- **Archivos WAV** originales
- **Metadata** temporal (días sin agua, momento de captura)
- **Multi-canal**: 4 canales de grabación simultánea

## Modelos Implementados

### 1. Clasificación de Validez (Objetivo Primario)
- **CNN para Espectrogramas**: ResNet, EfficientNet, Vision Transformer
- **Audio Transformers**: Audio Spectrogram Transformer (AST)
- **Hybrid CNN-RNN**: Para capturar patrones temporales
- **Multi-modal**: Combinando espectrogramas y features de audio

### 2. Análisis de Patrones Temporales (Objetivo Secundario)
- **Modelos de Serie Temporal**: LSTM, GRU para analizar evolución del estrés
- **Regresión**: Predicción de días sin agua
- **Clustering**: Identificación de patrones sonoros específicos
- **Análisis Espectral**: Frecuencias características por nivel de estrés

### 3. Traditional ML + Feature Engineering
- **XGBoost** con features extraídas de librosa
- **LightGBM** con optimización de hiperparámetros
- **SVM** con kernels avanzados
- **Random Forest** para interpretabilidad

### 4. Ensemble Methods
- **Voting Classifier** combinando mejores modelos
- **Stacking** con meta-learner
- **Blending** de predicciones

## Estructura del Proyecto

```
├── data/                          # Enlace simbólico al dataset
├── src/
│   ├── data/                      # Preprocessing y data loading
│   ├── models/                    # Implementación de modelos
│   ├── features/                  # Feature engineering
│   ├── evaluation/                # Métricas y evaluación
│   └── utils/                     # Utilidades
├── notebooks/                     # Jupyter notebooks para análisis
├── configs/                       # Configuraciones de modelos
├── experiments/                   # Resultados de experimentos
└── models/                        # Modelos entrenados
```

## Instalación

### Configuración con Conda (Recomendado)

```bash
# Ejecutar el script de setup automático
chmod +x setup.sh
./setup.sh

# O configuración manual:
conda create -n ibmcp python=3.9 -y
conda activate ibmcp

# Instalar PyTorch con soporte CUDA (si está disponible)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# O versión CPU: conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Instalar dependencias adicionales
pip install -r requirements.txt

# Crear enlace simbólico al dataset
ln -s /Volumes/RED2TB/ibmcp ./data
```

### Configuración con pip

```bash
pip install -r requirements.txt
```

## Uso

**Importante**: Activar el entorno conda antes de usar:
```bash
conda activate ibmcp
```

**Importante**: Activar el entorno conda antes de usar:
```bash
conda activate ibmcp
```

1. **Análisis Exploratorio**: `notebooks/01_exploratory_data_analysis.ipynb`
2. **Entrenamiento**: `python src/train.py --config configs/default_config.yaml`
3. **Evaluación**: `python src/evaluate.py --model_path models/best_model.pth`

## Framework y Tecnologías

- **PyTorch 2.1.0**: Framework principal de deep learning
- **Transformers**: Para Audio Spectrogram Transformer
- **TIMM**: Modelos pre-entrenados de computer vision
- **Librosa**: Procesamiento avanzado de audio
- **Optuna**: Optimización de hiperparámetros
- **MLflow**: Tracking de experimentos

## Resultados Esperados

### Clasificación de Validez (Primario)
- **Accuracy**: >90% (distinguir sonidos válidos vs ruido)
- **F1-Score**: >0.85 (balanceando precisión y recall)
- **AUC-ROC**: >0.95
- **Especificidad**: >0.90 (importante para reducir falsos positivos)

### Análisis Temporal (Secundario)
- **Correlación temporal**: Identificar patrones según días sin agua
- **Clustering accuracy**: Agrupar sonidos por nivel de estrés
- **Regresión R²**: >0.7 para predicción de días sin agua
- **Interpretabilidad**: Features más importantes para cada nivel de estrés

## Técnicas Avanzadas Implementadas

- Data augmentation específico para audio
- Transfer learning con modelos pre-entrenados
- Optimización de hiperparámetros con Optuna
- Ensemble learning y model stacking
- Class balancing avanzado
- Feature selection automática
