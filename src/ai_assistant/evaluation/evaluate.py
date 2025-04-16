import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import re
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Add parent directory to path to import app functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import calculate_differential_genes, visualize_gene_expression

class CellTypeEvaluator:
    def __init__(self, test_data_path, ground_truth_path):
        self.test_data = pd.read_csv(test_data_path)
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.results = []
        self.metrics = {}
        
    def evaluate_cell_type_prediction(self, predicted_cell_type, true_cell_type, cluster=None):
        """Evaluate a single cell type prediction"""
        return {
            'cluster': cluster,
            'predicted': predicted_cell_type,
            'true': true_cell_type,
            'correct': predicted_cell_type == true_cell_type
        }
    
    def calculate_metrics(self):
        """Calculate overall performance metrics"""
        if not self.results:
            return {}
            
        predictions = [r['predicted'] for r in self.results]
        truths = [r['true'] for r in self.results]
        
        self.metrics = {
            'accuracy': accuracy_score(truths, predictions),
            'precision': precision_score(truths, predictions, average='weighted', zero_division=0),
            'recall': recall_score(truths, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(truths, predictions, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(truths, predictions)
        }
        return self.metrics
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report = {
            'timestamp': timestamp,
            'total_samples': len(self.results),
            'results': self.results,
            'metrics': self.metrics,
            'confusion_matrix': self._generate_confusion_matrix()
        }
        
        # Save report to file
        os.makedirs(os.path.dirname('evaluation/reports/'), exist_ok=True)
        report_path = f'evaluation/reports/evaluation_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report_path
    
    def _generate_confusion_matrix(self):
        """Generate a confusion matrix for cell type predictions"""
        if not self.results:
            return {}
            
        # Get unique cell types
        true_types = set(r['true'] for r in self.results)
        pred_types = set(r['predicted'] for r in self.results)
        cell_types = sorted(list(true_types.union(pred_types)))
        
        # Initialize confusion matrix
        matrix = {t: {p: 0 for p in cell_types} for t in cell_types}
        
        # Fill in confusion matrix
        for result in self.results:
            true_type = result['true']
            pred_type = result['predicted']
            matrix[true_type][pred_type] += 1
        
        return matrix

def parse_llm_response(response_text):
    """Extract cell type from LLM response"""
    # First look for explicit statements
    explicit_patterns = [
        r"the cell type is ([\w\s-]+)",
        r"cluster represents ([\w\s-]+)",
        r"identified as ([\w\s-]+)",
        r"classified as ([\w\s-]+)"
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            return match.group(1).strip()
    
    # Look for common cell types
    cell_type_patterns = [
        r"(t-cell|b-cell|macrophage|neutrophil|nk cell|monocyte|dendritic cell)"
    ]
    
    for pattern in cell_type_patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            return match.group(1).strip()
    
    # If no match, return first line (which might contain the answer)
    first_line = response_text.split("\n")[0].strip()
    return first_line

def load_log_data(log_file):
    """Load evaluation logs from a JSONL file"""
    data = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    return data

def generate_report_from_logs(log_file):
    """Generate a comprehensive report from evaluation logs"""
    log_data = load_log_data(log_file)
    
    if not log_data:
        print("No log data found.")
        return None
    
    # Extract all predictions
    all_results = []
    for entry in log_data:
        if 'predictions' in entry:
            all_results.extend(entry['predictions'])
    
    # Calculate overall metrics
    evaluator = CellTypeEvaluator('sample_data.csv', 'ground_truth.csv')
    evaluator.results = all_results
    metrics = evaluator.calculate_metrics()
    
    # Generate report
    report_path = evaluator.generate_report()
    
    return report_path

def main():
    # Create directories if they don't exist
    os.makedirs('evaluation/reports', exist_ok=True)
    os.makedirs('evaluation/logs', exist_ok=True)
    
    # Check if we should generate a report from logs
    if len(sys.argv) > 1 and sys.argv[1] == '--from-logs':
        log_file = 'evaluation/logs/prediction_log.jsonl'
        if len(sys.argv) > 2:
            log_file = sys.argv[2]
        
        report_path = generate_report_from_logs(log_file)
        if report_path:
            print(f"Report generated from logs: {report_path}")
        return
    
    # Initialize evaluator with test data
    try:
        evaluator = CellTypeEvaluator(
            test_data_path='sample_data.csv',
            ground_truth_path='ground_truth.csv'
        )
        
        print("Data loaded successfully.")
        print(f"Test data shape: {evaluator.test_data.shape}")
        print(f"Ground truth data shape: {evaluator.ground_truth.shape}")
        
        # Run evaluation
        # For demonstration purposes, we'll use sample results
        sample_results = [
            {'cluster': 'cluster1', 'predicted': 'T-cell', 'true': 'T-cell', 'correct': True},
            {'cluster': 'cluster2', 'predicted': 'B-cell', 'true': 'B-cell', 'correct': True},
            {'cluster': 'cluster3', 'predicted': 'B-cell', 'true': 'B-cell', 'correct': True},
            {'cluster': 'cluster4', 'predicted': 'Neutrophil', 'true': 'Macrophage', 'correct': False}
        ]
        
        evaluator.results = sample_results
        metrics = evaluator.calculate_metrics()
        report_path = evaluator.generate_report()
        
        print(f"Evaluation complete! Report saved to: {report_path}")
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save to log file
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics,
            'query': "Sample evaluation run",
            'predictions': sample_results
        }
        
        log_file = 'evaluation/logs/prediction_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        print(f"\nResults logged to: {log_file}")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()