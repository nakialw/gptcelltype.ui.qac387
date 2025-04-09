import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from app import calculate_differential_genes, visualize_gene_expression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CellTypeEvaluator:
    def __init__(self, test_data_path, ground_truth_path):
        self.test_data = pd.read_csv(test_data_path)
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.results = []
        self.metrics = {}
        
    def evaluate_cell_type_prediction(self, predicted_cell_type, true_cell_type):
        """Evaluate a single cell type prediction"""
        return {
            'predicted': predicted_cell_type,
            'true': true_cell_type,
            'correct': predicted_cell_type == true_cell_type
        }
    
    def calculate_metrics(self):
        """Calculate overall performance metrics"""
        predictions = [r['predicted'] for r in self.results]
        truths = [r['true'] for r in self.results]
        
        self.metrics = {
            'accuracy': accuracy_score(truths, predictions),
            'precision': precision_score(truths, predictions, average='weighted'),
            'recall': recall_score(truths, predictions, average='weighted'),
            'f1_score': f1_score(truths, predictions, average='weighted')
        }
        return self.metrics
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report = {
            'timestamp': timestamp,
            'total_samples': len(self.results),
            'results': self.results,
            'metrics': self.metrics
        }
        
        # Save report to file
        report_path = f'evaluation/reports/evaluation_report_{timestamp}.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report_path

def main():
    # Initialize evaluator with test data
    evaluator = CellTypeEvaluator(
        test_data_path='sample_data.csv',
        ground_truth_path='ground_truth.csv'
    )
    
    # Run evaluation
    # Note: In a real scenario, you would run your model's predictions here
    # For now, we'll use a simple example
    sample_results = [
        {'predicted': 'T-cell', 'true': 'T-cell'},
        {'predicted': 'B-cell', 'true': 'B-cell'},
        {'predicted': 'Neutrophil', 'true': 'Macrophage'}
    ]
    
    evaluator.results = sample_results
    metrics = evaluator.calculate_metrics()
    report_path = evaluator.generate_report()
    
    print(f"Evaluation complete! Report saved to: {report_path}")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 