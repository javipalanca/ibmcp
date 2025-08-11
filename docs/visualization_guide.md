# Análisis de Ultrasonidos de Plantas - Visualización de Resultados

## Comando para Visualizar Resultados Comparativos

### Descripción
El script `plant_stress_analysis.py` ahora incluye un comando especial para visualizar y comparar los resultados de todos los modelos entrenados de forma gráfica e interactiva.

### Uso

```bash
# Visualizar resultados con directorio por defecto
python src/plant_stress_analysis.py --visualize-results

# Visualizar resultados de un directorio específico
python src/plant_stress_analysis.py --visualize-results --results-dir path/to/results

# Visualizar resultados para clasificación
python src/plant_stress_analysis.py --visualize-results --task-type classification
```

### Características de la Visualización

#### Para Regresión
- **Gráfico de barras R² Score**: Muestra el coeficiente de determinación (mayor = mejor)
- **Gráfico de barras RMSE**: Muestra el error cuadrático medio (menor = mejor)  
- **Gráfico de barras MAE**: Muestra el error absoluto medio (menor = mejor)
- **Radar Chart**: Comparación normalizada de todas las métricas
- **Ranking automático**: Clasificación ponderada de modelos

#### Para Clasificación
- **Gráfico de barras Accuracy**: Muestra la precisión de cada modelo
- **Gráfico de pizza**: Distribución visual de la precisión
- **Ranking por accuracy**: Ordenamiento de mejor a peor modelo

### Archivos Generados

1. **Gráficos PNG**: 
   - `model_comparison_regression.png` (para regresión)
   - `model_comparison_classification.png` (para clasificación)

2. **Estadísticas en consola**:
   - Ranking detallado de modelos
   - Mejor modelo por métrica
   - Estadísticas adicionales

### Requisitos

- Archivo `results_summary.json` debe existir en el directorio de resultados
- El archivo debe contener métricas de los modelos entrenados

### Ejemplo de Archivo de Resultados

```json
{
  "cnn_waveform": {
    "r2_score": 0.742,
    "rmse": 15.23,
    "mae": 12.45
  },
  "cnn_spectrogram": {
    "r2_score": 0.756,
    "rmse": 14.87,
    "mae": 11.98
  },
  "ensemble": {
    "r2_score": 0.812,
    "rmse": 13.12,
    "mae": 10.67
  }
}
```

### Interpretación de Resultados

#### Métricas de Regresión
- **R² Score (0-1)**: Proporción de varianza explicada. Valores más altos indican mejor ajuste.
- **RMSE**: Error cuadrático medio. Valores más bajos indican mejor predicción.
- **MAE**: Error absoluto medio. Valores más bajos indican mejor predicción.

#### Score Combinado
El ranking utiliza un score ponderado:
- R² Score: 50% del peso
- RMSE normalizado: 30% del peso  
- MAE normalizado: 20% del peso

### Workflow Completo

```bash
# 1. Entrenar modelos
python src/plant_stress_analysis.py --models all --epochs 50

# 2. Visualizar resultados
python src/plant_stress_analysis.py --visualize-results

# 3. Análisis adicional si es necesario
python src/plant_stress_analysis.py --analyze-only
```

### Resolución de Problemas

**Error: "No se encontró archivo de resultados"**
- Ejecuta primero el entrenamiento de modelos
- Verifica que el directorio de resultados sea correcto

**Error: "No hay resultados para visualizar"**
- El archivo `results_summary.json` está vacío
- Verifica que el entrenamiento se completó correctamente

**Gráficos no se muestran**
- En entornos sin GUI, los gráficos se guardan pero no se visualizan
- Revisa los archivos PNG generados en el directorio de resultados
