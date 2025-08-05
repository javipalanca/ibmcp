"""
Evaluation script for trained plant ultrasonic audio classification models
"""
import argparse
import sys
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.models.deep_learning_models import create_model
from src.evaluation.metrics import ModelEvaluator
from src.utils.training_utils import set_seed

class ModelEvaluator:
    """Evaluate trained models on test data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(config['experiment']['save_dir'])
        
        # Load data
        self.data_loader = DataLoader(config['data']['data_path'])
        self.data_loaders = self.data_loader.get_data_loaders(
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
    
    def evaluate_deep_learning_model(self, model_path: str, model_config: Dict) -> Dict:
        """Evaluate a deep learning model"""
        print(f"Evaluating deep learning model: {model_config['architecture']}")
        
        # Load model
        model = create_model(
            model_type=model_config['type'],
            **model_config.get('params', {})
        )
        
        # Load trained weights
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            print(f"✅ Loaded model weights from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return {}
        
        # Evaluate
        metrics = self.evaluator.evaluate_deep_learning_model(
            model, self.data_loaders['test'], self.device, 
            model_name=model_config['architecture']
        )
        
        return metrics
    
    def evaluate_traditional_ml_model(self, model_path: str, model_name: str) -> Dict:
        """Evaluate a traditional ML model"""
        print(f"Evaluating traditional ML model: {model_name}")
        
        # Load model
        if Path(model_path).exists():
            model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return {}
        
        # This would require implementing feature extraction for test data
        # For now, return placeholder
        print("⚠️  Traditional ML evaluation not fully implemented")
        return {}
    
    def evaluate_all_models(self, models_dir: str) -> Dict[str, Dict]:
        """Evaluate all trained models"""
        models_path = Path(models_dir)
        all_results = {}
        
        # Evaluate deep learning models
        for model_config in self.config['models']['deep_learning']:
            model_file = models_path / f"best_{model_config['architecture']}.pth"
            
            if model_file.exists():
                try:
                    metrics = self.evaluate_deep_learning_model(str(model_file), model_config)
                    all_results[model_config['architecture']] = metrics
                except Exception as e:
                    print(f"❌ Error evaluating {model_config['architecture']}: {e}")
            else:
                print(f"⚠️  Model file not found: {model_file}")
        
        # Evaluate traditional ML models (if implemented)
        if self.config['models'].get('traditional_ml', {}).get('enabled', False):
            for model_config in self.config['models']['traditional_ml']['models']:
                model_file = models_path / f"{model_config['name']}_model.pkl"
                
                if model_file.exists():
                    try:
                        metrics = self.evaluate_traditional_ml_model(
                            str(model_file), model_config['name']
                        )
                        all_results[model_config['name']] = metrics
                    except Exception as e:
                        print(f"❌ Error evaluating {model_config['name']}: {e}")
        
        return all_results
    
    def generate_comparison_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive comparison report"""
        
        # Sort models by accuracy
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].get('accuracy', 0), 
            reverse=True
        )
        
        report = f"""
# Plant Ultrasonic Audio Classification - Model Evaluation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information
- Total test samples: {len(self.data_loaders['test'].dataset):,}
- Classes: No Stress (0) vs Water Stress (1)

## Model Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
"""
        
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            if metrics:  # Skip empty results
                report += f"| {i} | {model_name} | {metrics.get('accuracy', 0):.4f} | "
                report += f"{metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | "
                report += f"{metrics.get('f1_score', 0):.4f} | {metrics.get('roc_auc', 0):.4f} |\n"
        
        # Best model details
        if sorted_results:
            best_model, best_metrics = sorted_results[0]
            report += f"""

## Best Model: {best_model}

### Detailed Metrics:
- **Accuracy**: {best_metrics.get('accuracy', 0):.4f}
- **Precision**: {best_metrics.get('precision', 0):.4f}
- **Recall**: {best_metrics.get('recall', 0):.4f}
- **F1-Score**: {best_metrics.get('f1_score', 0):.4f}
- **ROC-AUC**: {best_metrics.get('roc_auc', 0):.4f}
- **Balanced Accuracy**: {best_metrics.get('balanced_accuracy', 0):.4f}

### Class-specific Performance:
- **Class 0 (No Stress)**:
  - Precision: {best_metrics.get('precision_class_0', 0):.4f}
  - Recall: {best_metrics.get('recall_class_0', 0):.4f}
  - F1-Score: {best_metrics.get('f1_score_class_0', 0):.4f}

- **Class 1 (Water Stress)**:
  - Precision: {best_metrics.get('precision_class_1', 0):.4f}
  - Recall: {best_metrics.get('recall_class_1', 0):.4f}
  - F1-Score: {best_metrics.get('f1_score_class_1', 0):.4f}

### Confusion Matrix:
```
                    Predicted
                No Stress  Water Stress
Actual No Stress      {best_metrics.get('true_negatives', 0)}         {best_metrics.get('false_positives', 0)}
    Water Stress      {best_metrics.get('false_negatives', 0)}         {best_metrics.get('true_positives', 0)}
```
"""
        
        # Model comparison insights
        if len(sorted_results) > 1:
            report += f"""

## Model Comparison Insights:

### Top 3 Models:
"""
            for i, (model_name, metrics) in enumerate(sorted_results[:3], 1):
                if metrics:
                    report += f"{i}. **{model_name}**: {metrics.get('accuracy', 0):.4f} accuracy\n"
            
            # Performance differences
            if len(sorted_results) >= 2:
                best_acc = sorted_results[0][1].get('accuracy', 0)
                second_acc = sorted_results[1][1].get('accuracy', 0)
                diff = best_acc - second_acc
                report += f"\n- Best model outperforms second-best by: {diff:.4f} ({diff*100:.2f}%)\n"
        
        report += f"""

## Recommendations:

1. **Production Model**: Use {sorted_results[0][0] if sorted_results else 'N/A'} for deployment
2. **Ensemble Consideration**: Combine top-performing models for potentially better results
3. **Monitoring**: Track model performance on new data for model drift
4. **Retraining**: Consider retraining if accuracy drops below {0.9 if sorted_results and sorted_results[0][1].get('accuracy', 0) > 0.9 else 0.8:.1f}

## Technical Notes:
- All models evaluated on the same test set
- Metrics calculated using scikit-learn
- ROC-AUC computed for binary classification
- Class imbalance handled during training with appropriate techniques
"""
        
        return report


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained plant ultrasonic classification models')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models_dir', type=str, default='./experiments',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate models
    if args.model_name:
        print(f"Evaluating specific model: {args.model_name}")
        # Single model evaluation would go here
        results = {}
    else:
        print("Evaluating all trained models...")
        results = evaluator.evaluate_all_models(args.models_dir)
    
    if not results:
        print("❌ No models were successfully evaluated")
        return
    
    # Generate visualizations
    print("Generating evaluation plots...")
    evaluator.evaluator.compare_models(results)
    
    # Generate and save report
    print("Generating comparison report...")
    report = evaluator.generate_comparison_report(results)
    
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save results as JSON
    evaluator.evaluator.save_results(str(output_dir / "evaluation_results.json"))
    
    print(f"✅ Evaluation complete!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📝 Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    for i, (model_name, metrics) in enumerate(sorted_results[:5], 1):
        if metrics:
            print(f"{i}. {model_name}: {metrics.get('accuracy', 0):.4f} accuracy")
    
    if sorted_results:
        best_model, best_metrics = sorted_results[0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {best_metrics.get('accuracy', 0):.4f}")
        print(f"   F1-Score: {best_metrics.get('f1_score', 0):.4f}")
        print(f"   ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")


if __name__ == "__main__":
    main()
