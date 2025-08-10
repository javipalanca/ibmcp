#!/usr/bin/env python3
"""
Script de ejecución interactivo para entrenamiento de modelos de Deep Learning
Sistema completamente automatizado con detección de checkpoints y configuración fácil
"""

import subprocess
import sys
import json
import os
import torch
from pathlib import Path
from datetime import datetime

# Configuraciones por defecto
DEFAULT_CONFIG = {
    'epochs': 25,
    'batch_size': 16,
    'learning_rate': 0.001,
    'duration': 5.0,
    'save_every': 5,
    'sample_size': None,
    'device': 'auto'
}

# Información de modelos
MODELS_INFO = {
    'cnn1d': {
        'name': 'CNN 1D',
        'description': 'CNN para formas de onda directas',
        'input': 'Formas de onda',
        'tiempo_aprox': '~10 min/época',
        'recomendado_para': 'Análisis temporal directo'
    },
    'cnn2d': {
        'name': 'CNN 2D', 
        'description': 'CNN para espectrogramas mel',
        'input': 'Espectrogramas',
        'tiempo_aprox': '~15 min/época',
        'recomendado_para': 'Análisis frecuencial'
    },
    'resnet': {
        'name': 'ResNet Audio',
        'description': 'ResNet adaptada para audio',
        'input': 'Espectrogramas',
        'tiempo_aprox': '~20 min/época',
        'recomendado_para': 'Máxima precisión'
    },
    'lstm': {
        'name': 'LSTM',
        'description': 'LSTM para modelado temporal',
        'input': 'Secuencias temporales',
        'tiempo_aprox': '~25 min/época',
        'recomendado_para': 'Patrones temporales largos'
    },
    'gru': {
        'name': 'GRU',
        'description': 'GRU alternativo más rápido',
        'input': 'Secuencias temporales',
        'tiempo_aprox': '~20 min/época',
        'recomendado_para': 'Eficiencia temporal'
    },
    'hybrid': {
        'name': 'CNN-LSTM Híbrido',
        'description': 'Combina CNN y LSTM',
        'input': 'Formas de onda',
        'tiempo_aprox': '~30 min/época',
        'recomendado_para': 'Máximo rendimiento'
    }
}

def load_config():
    """Cargar configuración guardada si existe"""
    config_file = Path('dl_config.json')
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            # Combinar con defaults
            config = DEFAULT_CONFIG.copy()
            config.update(saved_config)
            return config
        except:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Guardar configuración actual"""
    config_file = Path('dl_config.json')
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass

def detect_checkpoints():
    """Detectar checkpoints existentes"""
    checkpoint_dir = Path('../checkpoints')
    if not checkpoint_dir.exists():
        return {}
    
    checkpoints = {}
    for model_dir in checkpoint_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            checkpoint_files = list(model_dir.glob('*_checkpoint_epoch_*.pt'))
            if checkpoint_files:
                # Ordenar por época (más reciente primero)
                checkpoint_files.sort(key=lambda x: int(x.name.split('_epoch_')[1].split('.')[0]), reverse=True)
                latest = checkpoint_files[0]
                epoch = int(latest.name.split('_epoch_')[1].split('.')[0])
                checkpoints[model_name] = {
                    'path': latest,
                    'epoch': epoch,
                    'file': latest.name,
                    'date': datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                }
    
    return checkpoints

def detect_existing_results():
    """Detectar modelos ya entrenados"""
    results_dir = Path('../models')
    if not results_dir.exists():
        return {}
    
    results = {}
    for results_file in results_dir.glob('*_results_*.json'):
        try:
            model_name = results_file.name.split('_results_')[0]
            timestamp = results_file.name.split('_results_')[1].replace('.json', '')
            date = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            results[model_name] = {
                'accuracy': data.get('accuracy', 0),
                'auc': data.get('auc', 0),
                'date': date,
                'file': results_file.name
            }
        except:
            continue
    
    return results

def print_model_info():
    """Mostrar información detallada de modelos"""
    print("\n📋 MODELOS DISPONIBLES:")
    print("="*80)
    
    for i, (model_id, info) in enumerate(MODELS_INFO.items(), 1):
        print(f"{i}. {info['name']} ({model_id})")
        print(f"   📝 {info['description']}")
        print(f"   📊 Entrada: {info['input']}")
        print(f"   ⏱️  Tiempo: {info['tiempo_aprox']}")
        print(f"   🎯 Recomendado para: {info['recomendado_para']}")
        print()

def select_models():
    """Seleccionar modelos interactivamente"""
    print_model_info()
    
    print("Opciones de selección:")
    print("0. Todos los modelos")
    print("7. Configuración personalizada")
    print("8. Solo evaluación de modelos existentes")
    
    while True:
        try:
            choice = input("\n>>> Selecciona opción (0-8): ").strip()
            
            if choice == "0":
                return list(MODELS_INFO.keys())
            elif choice == "8":
                return ["eval-only"]
            elif choice == "7":
                return select_custom_models()
            elif choice in ["1", "2", "3", "4", "5", "6"]:
                model_id = list(MODELS_INFO.keys())[int(choice) - 1]
                return [model_id]
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
        except (ValueError, IndexError):
            print("❌ Entrada inválida. Usa números del 0-8.")

def select_custom_models():
    """Selección personalizada de múltiples modelos"""
    print("\nSelección personalizada - Ingresa los números separados por comas (ej: 1,3,5)")
    print("O usa 'all' para todos los modelos")
    
    while True:
        selection = input(">>> Modelos: ").strip().lower()
        
        if selection == "all":
            return list(MODELS_INFO.keys())
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            models = []
            for idx in indices:
                if 1 <= idx <= len(MODELS_INFO):
                    model_id = list(MODELS_INFO.keys())[idx - 1]
                    models.append(model_id)
                else:
                    raise ValueError()
            
            if models:
                return models
            else:
                print("❌ No se seleccionaron modelos válidos.")
        except ValueError:
            print("❌ Formato inválido. Usa números separados por comas (ej: 1,3,5)")

def validate_config(config):
    """Validar configuración y mostrar advertencias"""
    warnings = []
    
    # Validar duración mínima
    if config['duration'] < 2.0:
        warnings.append(f"⚠️  Duración muy corta ({config['duration']}s). Recomendado: 3-10s")
    
    # Validar batch size vs GPU
    if config['device'] == 'cuda' and config['batch_size'] > 128:
        warnings.append(f"⚠️  Batch size muy grande ({config['batch_size']}). Podría causar error de memoria")
    
    # Validar learning rate
    if config['learning_rate'] > 0.01:
        warnings.append(f"⚠️  Learning rate alto ({config['learning_rate']}). Podría causar inestabilidad")
    elif config['learning_rate'] < 1e-5:
        warnings.append(f"⚠️  Learning rate muy bajo ({config['learning_rate']}). Entrenamiento muy lento")
    
    # Validar épocas vs muestra
    if config['sample_size'] and config['sample_size'] < 1000 and config['epochs'] > 20:
        warnings.append(f"⚠️  Muchas épocas ({config['epochs']}) para muestra pequeña ({config['sample_size']})")
    
    return warnings

def configure_hyperparameters(current_config):
    """Configurar hiperparámetros interactivamente"""
    print("\n⚙️  CONFIGURACIÓN DE HIPERPARÁMETROS")
    print("="*50)
    print("Presiona Enter para mantener el valor actual")
    
    config = current_config.copy()
    
    # Épocas
    print(f"\n🔢 Épocas (actual: {config['epochs']})")
    print("   Recomendado: 5-10 para pruebas, 25-50 para producción")
    new_epochs = input(">>> Épocas: ").strip()
    if new_epochs:
        try:
            config['epochs'] = int(new_epochs)
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Batch size
    print(f"\n📦 Batch Size (actual: {config['batch_size']})")
    print("   Recomendado: 8-16 para GPU con poca memoria, 32-64 para GPU potente")
    new_batch = input(">>> Batch size: ").strip()
    if new_batch:
        try:
            config['batch_size'] = int(new_batch)
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Learning rate
    print(f"\n📈 Learning Rate (actual: {config['learning_rate']})")
    print("   Recomendado: 0.001 estándar, 0.0001 para ajuste fino, 0.01 para convergencia rápida")
    new_lr = input(">>> Learning rate: ").strip()
    if new_lr:
        try:
            config['learning_rate'] = float(new_lr)
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Duración de audio
    print(f"\n🎵 Duración de audio en segundos (actual: {config['duration']})")
    print("   Recomendado: 3-10 segundos para balance tiempo/información")
    print("   MÍNIMO: 2 segundos (duraciones menores causan errores)")
    new_duration = input(">>> Duración: ").strip()
    if new_duration:
        try:
            duration = float(new_duration)
            if duration < 1.0:
                print("⚠️  Duración demasiado corta, estableciendo mínimo de 2s")
                config['duration'] = 2.0
            else:
                config['duration'] = duration
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Tamaño de muestra
    current_sample = config['sample_size'] or "Completo"
    print(f"\n🎯 Tamaño de muestra (actual: {current_sample})")
    print("   Ejemplos: 100 para pruebas rápidas, 1000 para validación, vacío para dataset completo")
    new_sample = input(">>> Tamaño muestra: ").strip()
    if new_sample:
        try:
            if new_sample.lower() in ["completo", "full", ""]:
                config['sample_size'] = None
            else:
                sample_size = int(new_sample)
                if sample_size < 50:
                    print("⚠️  Muestra muy pequeña, estableciendo mínimo de 50")
                    config['sample_size'] = 50
                else:
                    config['sample_size'] = sample_size
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Frecuencia de guardado
    print(f"\n💾 Guardar checkpoint cada N épocas (actual: {config['save_every']})")
    print("   Recomendado: 5 para entrenamientos largos, 2-3 para monitoreo frecuente")
    new_save = input(">>> Frecuencia guardado: ").strip()
    if new_save:
        try:
            config['save_every'] = max(1, int(new_save))
        except ValueError:
            print("⚠️  Valor inválido, manteniendo actual")
    
    # Dispositivo
    print(f"\n🖥️  Dispositivo (actual: {config['device']})")
    print("   Opciones: auto (detectar automáticamente), cpu, cuda")
    new_device = input(">>> Dispositivo: ").strip().lower()
    if new_device and new_device in ['auto', 'cpu', 'cuda']:
        config['device'] = new_device
    elif new_device:
        print("⚠️  Dispositivo inválido, manteniendo actual")
    
    # Validar configuración
    warnings = validate_config(config)
    if warnings:
        print(f"\n⚠️  ADVERTENCIAS DE CONFIGURACIÓN:")
        for warning in warnings:
            print(f"   {warning}")
        
        fix_config = input("\n¿Aplicar correcciones automáticas? (Y/n): ").strip().lower()
        if fix_config != 'n':
            config = apply_auto_corrections(config)
    
    return config

def apply_auto_corrections(config):
    """Aplicar correcciones automáticas a la configuración"""
    print("\n🔧 Aplicando correcciones automáticas...")
    
    # Corregir duración mínima
    if config['duration'] < 2.0:
        config['duration'] = 3.0
        print(f"   ✅ Duración corregida a {config['duration']}s")
    
    # Corregir batch size para GPU
    if config['device'] == 'cuda' and config['batch_size'] > 128:
        config['batch_size'] = 32
        print(f"   ✅ Batch size corregido a {config['batch_size']}")
    
    # Corregir learning rate
    if config['learning_rate'] > 0.01:
        config['learning_rate'] = 0.001
        print(f"   ✅ Learning rate corregido a {config['learning_rate']}")
    elif config['learning_rate'] < 1e-5:
        config['learning_rate'] = 1e-4
        print(f"   ✅ Learning rate corregido a {config['learning_rate']}")
    
    # Corregir épocas vs muestra
    if config['sample_size'] and config['sample_size'] < 1000 and config['epochs'] > 20:
        config['epochs'] = 10
        print(f"   ✅ Épocas corregidas a {config['epochs']} para muestra pequeña")
    
    return config

def handle_checkpoints(models, checkpoints):
    """Manejar checkpoints existentes"""
    if not checkpoints:
        return False, []
    
    # Filtrar checkpoints de modelos seleccionados
    relevant_checkpoints = {k: v for k, v in checkpoints.items() if k in models}
    
    if not relevant_checkpoints:
        return False, []
    
    print("\n🔄 CHECKPOINTS DETECTADOS")
    print("="*40)
    for model, info in relevant_checkpoints.items():
        print(f"📁 {model}: Época {info['epoch']} ({info['date']})")
    
    print("\nOpciones:")
    print("1. Continuar desde checkpoints")
    print("2. Empezar desde cero")
    print("3. Ver detalles de checkpoints")
    
    while True:
        choice = input(">>> Opción (1-3): ").strip()
        
        if choice == "1":
            return True, relevant_checkpoints
        elif choice == "2":
            # Preguntar si eliminar checkpoints
            delete = input("¿Eliminar checkpoints existentes? (y/N): ").strip().lower()
            if delete == 'y':
                for info in relevant_checkpoints.values():
                    try:
                        info['path'].unlink()
                        print(f"🗑️  Eliminado: {info['file']}")
                    except:
                        pass
            return False, []
        elif choice == "3":
            show_checkpoint_details(relevant_checkpoints)
        else:
            print("❌ Opción no válida.")

def show_checkpoint_details(checkpoints):
    """Mostrar detalles de checkpoints"""
    print("\n📊 DETALLES DE CHECKPOINTS")
    print("="*50)
    
    for model, info in checkpoints.items():
        print(f"\n🎯 {model.upper()}")
        print(f"   📄 Archivo: {info['file']}")
        print(f"   📅 Fecha: {info['date']}")
        print(f"   🔢 Época: {info['epoch']}")
        print(f"   📍 Ruta: {info['path']}")
        
        # Intentar leer métricas del checkpoint si es posible
        try:
            import torch
            checkpoint_data = torch.load(info['path'], map_location='cpu')
            if 'metrics' in checkpoint_data:
                metrics = checkpoint_data['metrics']
                if 'best_val_acc' in metrics:
                    print(f"   🎯 Mejor accuracy: {metrics['best_val_acc']:.2f}%")
        except:
            pass

def show_existing_results(results):
    """Mostrar resultados de modelos ya entrenados"""
    if not results:
        print("\n📊 No hay modelos entrenados previamente")
        return
    
    print("\n🏆 MODELOS YA ENTRENADOS")
    print("="*50)
    
    # Ordenar por AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for model, info in sorted_results:
        print(f"\n🎯 {model.upper()}")
        print(f"   🎯 Accuracy: {info['accuracy']:.2f}%")
        print(f"   📈 AUC: {info['auc']:.3f}")
        print(f"   📅 Fecha: {info['date']}")

def estimate_time(models, config):
    """Estimar tiempo total de entrenamiento"""
    time_per_epoch = {
        'cnn1d': 10,
        'cnn2d': 15, 
        'resnet': 20,
        'lstm': 25,
        'gru': 20,
        'hybrid': 30
    }
    
    total_minutes = 0
    epochs = config['epochs']
    
    for model in models:
        if model in time_per_epoch:
            total_minutes += time_per_epoch[model] * epochs
    
    # Ajustar por tamaño de muestra
    if config['sample_size']:
        # Escalado aproximado
        factor = min(1.0, config['sample_size'] / 10000)
        total_minutes *= factor
    
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    return hours, minutes

def print_summary(models, config, use_checkpoints, time_estimate):
    """Mostrar resumen de la configuración"""
    print("\n📋 RESUMEN DE CONFIGURACIÓN")
    print("="*50)
    
    # Modelos
    if models == ["eval-only"]:
        print("🎯 Modo: Solo evaluación")
    else:
        print(f"🎯 Modelos: {', '.join(models)}")
    
    # Configuración
    print(f"🔢 Épocas: {config['epochs']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"📈 Learning rate: {config['learning_rate']}")
    print(f"🎵 Duración audio: {config['duration']}s")
    
    sample_text = f"{config['sample_size']:,}" if config['sample_size'] else "Completo"
    print(f"🎯 Tamaño muestra: {sample_text}")
    
    print(f"💾 Checkpoint cada: {config['save_every']} épocas")
    print(f"🖥️  Dispositivo: {config['device']}")
    
    # Checkpoints
    if use_checkpoints:
        print("🔄 Continuar desde checkpoints: Sí")
    
    # Tiempo estimado
    if models != ["eval-only"] and time_estimate[0] > 0 or time_estimate[1] > 0:
        if time_estimate[0] > 0:
            print(f"⏱️  Tiempo estimado: {time_estimate[0]}h {time_estimate[1]}min")
        else:
            print(f"⏱️  Tiempo estimado: {time_estimate[1]}min")

def build_command(models, config, use_checkpoints, checkpoints):
    """Construir comando de ejecución"""
    cmd = [sys.executable, "deep_learning_training.py"]
    
    # Modelos
    if models == ["eval-only"]:
        cmd.append("--eval-only")
    else:
        cmd.extend(["--models"] + models)
    
    # Configuración
    cmd.extend([
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--learning-rate", str(config['learning_rate']),
        "--duration", str(config['duration']),
        "--save-every", str(config['save_every']),
        "--device", config['device']
    ])
    
    if config['sample_size']:
        cmd.extend(["--sample-size", str(config['sample_size'])])
    
    # Añadir opción para continuar desde checkpoints
    if use_checkpoints:
        cmd.append("--resume-from-checkpoints")
    
    # Añadir estrategia de GPU si es relevante
    if torch.cuda.is_available():
        cmd.extend(["--gpu-strategy", "optimal"])
    
    return cmd

def main():
    """Función principal completamente interactiva"""
    
    print("🎵 SISTEMA INTERACTIVO DE DEEP LEARNING")
    print("="*60)
    print("🤖 Configuración automática y detección inteligente")
    
    # Mostrar información de GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 GPUs detectadas: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
    else:
        print("🔧 No hay GPU disponible - usando CPU")
    
    # Cargar configuración previa
    config = load_config()
    
    # Detectar estado actual
    print("\n🔍 Detectando estado del sistema...")
    checkpoints = detect_checkpoints()
    existing_results = detect_existing_results()
    
    if checkpoints:
        print(f"✅ Detectados {len(checkpoints)} checkpoints")
    if existing_results:
        print(f"✅ Detectados {len(existing_results)} modelos entrenados")
    
    # Mostrar resultados existentes si los hay
    if existing_results:
        show_existing_results(existing_results)
    
    print("\n" + "="*60)
    
    # Menú principal
    while True:
        print("\n🎯 ¿QUÉ QUIERES HACER?")
        print("1. 🚀 Entrenar modelo(s)")
        print("2. ⚙️  Configurar hiperparámetros")
        print("3. 📊 Solo evaluar modelos existentes")
        print("4. 🗑️  Limpiar checkpoints/resultados")
        print("5. ❓ Ayuda y ejemplos")
        print("0. 🚪 Salir")
        
        choice = input("\n>>> Opción: ").strip()
        
        if choice == "0":
            print("👋 ¡Hasta luego!")
            break
            
        elif choice == "1":
            # Entrenar modelos
            train_workflow(config, checkpoints, existing_results)
            break
            
        elif choice == "2":
            # Configurar hiperparámetros
            config = configure_hyperparameters(config)
            save_config(config)
            print("\n✅ Configuración guardada")
            
        elif choice == "3":
            # Solo evaluación
            if existing_results:
                confirm = input("\n¿Ejecutar evaluación de modelos existentes? (Y/n): ").strip().lower()
                if confirm != 'n':
                    run_evaluation()
                    break
            else:
                print("\n❌ No hay modelos entrenados para evaluar")
                
        elif choice == "4":
            # Limpiar datos
            cleanup_menu(checkpoints, existing_results)
            # Redetectar después de limpieza
            checkpoints = detect_checkpoints()
            existing_results = detect_existing_results()
            
        elif choice == "5":
            # Ayuda
            show_help()
            
        else:
            print("❌ Opción no válida")

def train_workflow(config, checkpoints, existing_results):
    """Flujo completo de entrenamiento"""
    
    # 1. Seleccionar modelos
    print("\n" + "="*60)
    print("PASO 1: SELECCIÓN DE MODELOS")
    models = select_models()
    
    if models == ["eval-only"]:
        run_evaluation()
        return
    
    # 2. Configurar hiperparámetros
    print("\n" + "="*60)
    print("PASO 2: CONFIGURACIÓN")
    
    print(f"\nConfiguración actual:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    change_config = input("\n¿Cambiar configuración? (y/N): ").strip().lower()
    if change_config == 'y':
        config = configure_hyperparameters(config)
        save_config(config)
    
    # 3. Manejar checkpoints
    print("\n" + "="*60)
    print("PASO 3: CHECKPOINTS")
    
    use_checkpoints = False
    relevant_checkpoints = {}
    
    if checkpoints:
        use_checkpoints, relevant_checkpoints = handle_checkpoints(models, checkpoints)
    else:
        print("ℹ️  No hay checkpoints existentes. Empezando desde cero.")
    
    # 4. Estimar tiempo
    time_estimate = estimate_time(models, config)
    
    # 5. Mostrar resumen y confirmar
    print("\n" + "="*60)
    print("PASO 4: CONFIRMACIÓN")
    
    print_summary(models, config, use_checkpoints, time_estimate)
    
    # 6. Confirmar ejecución
    print("\n🎯 ¿PROCEDER CON EL ENTRENAMIENTO?")
    
    if time_estimate[0] > 2:  # Más de 2 horas
        print("⚠️  ATENCIÓN: El entrenamiento tomará mucho tiempo")
        print("   Recomendación: Ejecutar en screen/tmux o usar muestra más pequeña")
    
    confirm = input("\n¿Continuar? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("❌ Entrenamiento cancelado")
        return
    
    # 7. Ejecutar
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO")
    
    cmd = build_command(models, config, use_checkpoints, relevant_checkpoints)
    print(f"📋 Comando: {' '.join(cmd)}")
    
    # Mostrar tip para interrumpir
    print("\n💡 TIPS:")
    print("   - Ctrl+C para interrumpir (guarda checkpoint automático)")
    print("   - Los checkpoints se guardan automáticamente")
    print("   - Resultados se guardan en ../models/")
    
    input("\nPresiona Enter para comenzar...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por el usuario")

def cleanup_menu(checkpoints, existing_results):
    """Menú de limpieza"""
    print("\n🗑️  LIMPIEZA DE DATOS")
    print("="*30)
    
    print("¿Qué quieres limpiar?")
    print("1. Checkpoints")
    print("2. Resultados de modelos")
    print("3. Todo")
    print("0. Cancelar")
    
    choice = input("\n>>> Opción: ").strip()
    
    if choice == "1" and checkpoints:
        cleanup_checkpoints(checkpoints)
    elif choice == "2" and existing_results:
        cleanup_results(existing_results)
    elif choice == "3":
        if checkpoints:
            cleanup_checkpoints(checkpoints)
        if existing_results:
            cleanup_results(existing_results)
        cleanup_config()
    elif choice == "0":
        return
    else:
        print("❌ Opción no válida o no hay datos para limpiar")

def cleanup_checkpoints(checkpoints):
    """Limpiar checkpoints"""
    print(f"\n🗑️  Limpiando {len(checkpoints)} checkpoints...")
    
    for model, info in checkpoints.items():
        try:
            # Eliminar archivo de checkpoint
            info['path'].unlink()
            print(f"✅ Eliminado checkpoint de {model}")
            
            # Eliminar directorio si está vacío
            if not any(info['path'].parent.iterdir()):
                info['path'].parent.rmdir()
                
        except Exception as e:
            print(f"❌ Error eliminando {model}: {e}")

def cleanup_results(existing_results):
    """Limpiar resultados"""
    print(f"\n🗑️  Limpiando {len(existing_results)} resultados...")
    
    results_dir = Path('../models')
    for model, info in existing_results.items():
        try:
            # Buscar y eliminar archivos relacionados
            pattern = f"{model}_*"
            for file_path in results_dir.glob(pattern):
                file_path.unlink()
                print(f"✅ Eliminado {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error eliminando archivos de {model}: {e}")

def cleanup_config():
    """Limpiar configuración"""
    config_file = Path('dl_config.json')
    if config_file.exists():
        config_file.unlink()
        print("✅ Configuración eliminada")

def run_evaluation():
    """Ejecutar solo evaluación"""
    cmd = [sys.executable, "deep_learning_training.py", "--eval-only"]
    print("📊 Ejecutando evaluación de modelos existentes...")
    print(f"Comando: {' '.join(cmd)}")
    subprocess.run(cmd)

def show_help():
    """Mostrar ayuda detallada"""
    print("\n❓ AYUDA DEL SISTEMA")
    print("="*50)
    
    print("\n🎯 FLUJO TÍPICO:")
    print("1. Configurar hiperparámetros (una vez)")
    print("2. Seleccionar modelo(s) a entrenar") 
    print("3. El sistema detecta checkpoints automáticamente")
    print("4. Elegir continuar o empezar desde cero")
    print("5. Revisar resumen y confirmar")
    print("6. ¡Entrenar!")
    
    print("\n💡 RECOMENDACIONES:")
    print("🔸 Para pruebas: 1 modelo, muestra 100-500, 5 épocas")
    print("🔸 Para validación: 2-3 modelos, muestra 1000, 15 épocas")
    print("🔸 Para producción: Todos modelos, dataset completo, 25-50 épocas")
    
    print("\n⚡ OPTIMIZACIÓN:")
    print("🔸 GPU recomendada para entrenamiento")
    print("🔸 Usar batch_size más grande si tienes memoria GPU")
    print("🔸 Los checkpoints permiten reanudar entrenamiento")
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("🔸 ../checkpoints/: Checkpoints por modelo")
    print("🔸 ../models/: Modelos finales y resultados")
    print("🔸 dl_config.json: Tu configuración guardada")
    
    print("\n🆘 SOLUCIÓN DE PROBLEMAS:")
    print("🔸 Error de memoria: Reducir batch_size")
    print("🔸 Entrenamiento lento: Usar GPU o muestra más pequeña")
    print("🔸 Interrumpir: Ctrl+C (guarda checkpoint automático)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("💡 Intenta de nuevo o reporta el problema")
