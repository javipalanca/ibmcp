"""
Comprehensive evaluation metrics and utilities for model assessment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import joblib
import time

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, save_dir: str = "./experiments"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def evaluate_binary_classifier(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None,
                                 model_name: str = "model") -> Dict[str, float]:
        """Comprehensive evaluation for binary classification"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Class-specific metrics
        metrics['precision_class_0'] = precision_score(y_true, y_pred, pos_label=0)
        metrics['precision_class_1'] = precision_score(y_true, y_pred, pos_label=1)
        metrics['recall_class_0'] = recall_score(y_true, y_pred, pos_label=0)
        metrics['recall_class_1'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['f1_score_class_0'] = f1_score(y_true, y_pred, pos_label=0)
        metrics['f1_score_class_1'] = f1_score(y_true, y_pred, pos_label=1)
        
        # Probability-based metrics
        if y_prob is not None:
            if y_prob.ndim == 2:
                y_prob_pos = y_prob[:, 1]  # Probability of positive class
            else:
                y_prob_pos = y_prob
            
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_pos)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob_pos)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, 
                           model: Any, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv_folds: int = 5,
                           scoring: List[str] = None,
                           model_name: str = "model") -> Dict[str, Dict[str, float]]:
        """Perform cross-validation with multiple metrics"""
        
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 
                      'f1_weighted', 'roc_auc']
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=skf, scoring=score, n_jobs=-1)
            cv_results[score] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        
        # Store cross-validation results
        self.results[f"{model_name}_cv"] = cv_results
        
        return cv_results
    
    def evaluate_deep_learning_model(self, 
                                   model: torch.nn.Module, 
                                   data_loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   model_name: str = "dl_model") -> Dict[str, float]:
        """Evaluate PyTorch deep learning model"""
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    # Multi-modal model
                    if 'image' in batch and 'audio' in batch:
                        outputs = model(batch['image'].to(device), batch['audio'].to(device))
                    else:
                        outputs = model(batch['image'].to(device))
                    labels = batch['label'].to(device)
                else:
                    # Single input model
                    inputs, labels = batch
                    outputs = model(inputs.to(device))
                    labels = labels.to(device)
                
                # Get probabilities and predictions
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Evaluate
        metrics = self.evaluate_binary_classifier(y_true, y_pred, y_prob, model_name)
        
        return metrics
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            model_name: str = "model",
                            save_plot: bool = True) -> plt.Figure:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Stress', 'Water Stress'],
                   yticklabels=['No Stress', 'Water Stress'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_plot:
            plt.savefig(self.save_dir / f'confusion_matrix_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_prob: np.ndarray,
                      model_name: str = "model",
                      save_plot: bool = True) -> plt.Figure:
        """Plot ROC curve"""
        
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]  # Probability of positive class
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(self.save_dir / f'roc_curve_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_prob: np.ndarray,
                                   model_name: str = "model",
                                   save_plot: bool = True) -> plt.Figure:
        """Plot Precision-Recall curve"""
        
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]  # Probability of positive class
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(self.save_dir / f'pr_curve_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def compare_models(self, 
                      models_results: Dict[str, Dict[str, float]],
                      metrics_to_compare: List[str] = None,
                      save_plot: bool = True) -> plt.Figure:
        """Compare multiple models across different metrics"""
        
        if metrics_to_compare is None:
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Prepare data for plotting
        model_names = list(models_results.keys())
        comparison_data = []
        
        for metric in metrics_to_compare:
            metric_values = []
            for model_name in model_names:
                if metric in models_results[model_name]:
                    metric_values.append(models_results[model_name][metric])
                else:
                    metric_values.append(0)  # Default value if metric not found
            comparison_data.append(metric_values)
        
        # Create comparison plot
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for i, (metric, values) in enumerate(zip(metrics_to_compare, comparison_data)):
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(metrics_to_compare) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.save_dir / 'model_comparison.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_detailed_report(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               model_name: str = "model",
                               save_report: bool = True) -> str:
        """Generate detailed classification report"""
        
        report = classification_report(y_true, y_pred, 
                                     target_names=['No Stress', 'Water Stress'],
                                     output_dict=False)
        
        # Add confusion matrix information
        cm = confusion_matrix(y_true, y_pred)
        
        detailed_report = f"""
# Detailed Classification Report - {model_name}

## Classification Report
```
{report}
```

## Confusion Matrix
```
                    Predicted
                No Stress  Water Stress
Actual No Stress      {cm[0,0]}         {cm[0,1]}
    Water Stress      {cm[1,0]}         {cm[1,1]}
```

## Additional Metrics
- Balanced Accuracy: {self.results.get(model_name, {}).get('balanced_accuracy', 'N/A')}
- Specificity: {self.results.get(model_name, {}).get('specificity', 'N/A')}
- Sensitivity: {self.results.get(model_name, {}).get('sensitivity', 'N/A')}

## Class Distribution
- No Stress (Class 0): {np.sum(y_true == 0)} samples ({np.mean(y_true == 0):.2%})
- Water Stress (Class 1): {np.sum(y_true == 1)} samples ({np.mean(y_true == 1):.2%})
"""
        
        if save_report:
            with open(self.save_dir / f'detailed_report_{model_name}.md', 'w') as f:
                f.write(detailed_report)
        
        return detailed_report
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save all evaluation results to JSON file"""
        
        results_file = self.save_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_results[model_name][key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[model_name][key] = float(value)
                else:
                    serializable_results[model_name][key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def load_results(self, filename: str = "evaluation_results.json"):
        """Load evaluation results from JSON file"""
        
        results_file = self.save_dir / filename
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            print(f"Results loaded from {results_file}")
        else:
            print(f"Results file {results_file} not found")


class ExperimentTracker:
    """Track and manage machine learning experiments"""
    
    def __init__(self, experiment_dir: str = "./experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, 
                        experiment_name: str, 
                        model_config: Dict,
                        data_config: Dict = None) -> str:
        """Start a new experiment"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        experiment_path = self.experiment_dir / experiment_id
        experiment_path.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config = {
            'experiment_name': experiment_name,
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_config': model_config,
            'data_config': data_config or {}
        }
        
        with open(experiment_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        self.current_experiment = {
            'id': experiment_id,
            'path': experiment_path,
            'config': config
        }
        
        print(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current experiment"""
        
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        metrics_file = self.current_experiment['path'] / 'metrics.json'
        
        # Load existing metrics
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Add new metrics
        metric_entry = {'step': step, 'timestamp': time.time(), **metrics}
        all_metrics.append(metric_entry)
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def save_model(self, model: Any, model_name: str = "model"):
        """Save model to current experiment directory"""
        
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        model_path = self.current_experiment['path'] / f"{model_name}.pkl"
        
        # Save model using joblib
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def finish_experiment(self):
        """Finish current experiment"""
        
        if self.current_experiment:
            print(f"Finished experiment: {self.current_experiment['id']}")
            self.current_experiment = None


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Generate sample data for testing
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
    y_pred = np.random.choice([0, 1], size=1000, p=[0.65, 0.35])
    y_prob = np.random.random((1000, 2))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Evaluate model
    metrics = evaluator.evaluate_binary_classifier(y_true, y_pred, y_prob, "test_model")
    print("Test metrics:", metrics)
    
    # Create plots
    evaluator.plot_confusion_matrix(y_true, y_pred, "test_model")
    evaluator.plot_roc_curve(y_true, y_prob, "test_model")
    evaluator.plot_precision_recall_curve(y_true, y_prob, "test_model")
    
    # Generate report
    report = evaluator.generate_detailed_report(y_true, y_pred, "test_model")
    print("Generated detailed report")
    
    # Save results
    evaluator.save_results()
    
    print("Evaluation complete!")
