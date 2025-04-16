# Cell Type Annotation Evaluation Framework

This directory contains tools for evaluating the performance of the Cell Type Annotation Assistant.

## Directory Structure
```
evaluation/
├── evaluate.py         # Main evaluation script
├── logs/               # Directory for evaluation logs (JSONL format)
├── reports/            # Directory for evaluation reports (JSON format)
└── README.md           # This file
```

## Evaluation Metrics

The framework calculates the following metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among all positive predictions
- **Recall**: Proportion of true positives among all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement between predictions and ground truth accounting for chance agreement

## How to Use

### Standalone Evaluation
1. **Prepare Your Data**:
   - Place your test data in `sample_data.csv`
   - Create a `ground_truth.csv` file with known cell type annotations

2. **Run the Evaluation**:
   ```bash
   python evaluate.py
   ```

3. **View Results**:
   - Evaluation reports are saved in the `reports/` directory
   - Each report is timestamped and contains:
     - Overall metrics
     - Individual prediction results
     - Timestamp of evaluation

### In-App Evaluation
1. Navigate to the "Evaluation" tab in the Streamlit application
2. Upload your ground truth data
3. Select a cluster to evaluate
4. Run the prediction and evaluation
5. View results and metrics in the application
6. Review prediction history for past evaluations

## Log Format
The application logs evaluation metrics to `logs/prediction_log.jsonl` in the following format:

```json
{
    "timestamp": "2025-04-15 14:30:00",
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.86,
        "f1_score": 0.85,
        "cohen_kappa": 0.83
    },
    "query": "What cell type is most likely represented by cluster1?",
    "predictions": [
        {
            "cluster": "cluster1",
            "predicted": "T-cell",
            "true": "T-cell",
            "correct": true
        }
    ]
}
```

## Generating Evaluation Reports

To generate a comprehensive evaluation report from logs:

```python
from evaluation.evaluate import generate_report_from_logs

# Generate a report from all logs
report_path = generate_report_from_logs('evaluation/logs/prediction_log.jsonl')
print(f"Report generated at: {report_path}")
```

## Adding to GitHub

To add evaluation results to your GitHub repository:

1. Commit the evaluation logs and reports:
   ```bash
   git add evaluation/logs/ evaluation/reports/
   git commit -m "Add evaluation results"
   git push
   ```

2. Update the main README.md to include a summary of the latest evaluation metrics.