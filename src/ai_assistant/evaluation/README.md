# Cell Type Annotation Evaluation Framework

This directory contains tools for evaluating the performance of the Cell Type Annotation Assistant.

## Directory Structure
```
evaluation/
├── evaluate.py          # Main evaluation script
├── reports/            # Directory for evaluation reports
└── README.md           # This file
```

## Evaluation Metrics

The framework calculates the following metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among all positive predictions
- **Recall**: Proportion of true positives among all actual positives
- **F1 Score**: Harmonic mean of precision and recall

## How to Use

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

## Example Report Format
```json
{
    "timestamp": "2024-04-09_14-30-00",
    "total_samples": 100,
    "results": [
        {
            "predicted": "T-cell",
            "true": "T-cell",
            "correct": true
        },
        ...
    ],
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.86,
        "f1_score": 0.85
    }
}
```

## Adding to GitHub

To add evaluation results to your GitHub repository:

1. Commit the evaluation reports:
   ```bash
   git add evaluation/reports/
   git commit -m "Add evaluation results"
   git push
   ```

2. Update the main README.md to include a summary of the latest evaluation results. 