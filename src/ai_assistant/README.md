# Cell Type Annotation AI Assistant

This is an AI-powered assistant designed to help with cell type annotation in single-cell RNA sequencing analysis. It's part of the larger GPTCelltype enhancement project with integrations for VisCello-like visualizations.

## Features

- Interactive data upload and analysis
- Natural language queries about cell types and marker genes
- AI-powered analysis and recommendations with GPT
- Advanced visualizations including UMAP, heatmaps, and violin plots
- Integration with GPTCelltype and VisCello workflows
- Scanpy-based clustering analysis
- Performance evaluation with accuracy, precision, recall, F1, and Cohen's kappa

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload your single-cell data (CSV format with gene names in the first column)
3. Choose from multiple analysis options:
   - **Data Analysis**: Identify cell types, explore marker genes, interpret differential expression
   - **Visualization**: Generate UMAP plots, heatmaps, and compare clusters
   - **Clustering**: Perform clustering analysis with scanpy
   - **Evaluation**: Compare predictions against ground truth data

### Analysis Features

- Query templates for common analysis tasks
- Interactive visualizations for gene expression patterns
- AI-powered interpretation of visualizations
- Clustering and dimensional reduction
- Evaluation metrics tracking and reporting

## Evaluation

The assistant's performance is evaluated based on:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall**: Proportion of true positives among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement between predictions and ground truth

Evaluation results are saved to `evaluation/logs/prediction_log.jsonl` for tracking performance over time.

## Integration with GPTCelltype and VisCello

This assistant is designed to complement these packages by:
1. Providing intuitive interfaces for preliminary analysis
2. Helping interpret results with AI assistance
3. Offering VisCello-like visualizations for data exploration
4. Integrating scanpy-based clustering for cell type identification
5. Supporting evaluation against known annotations

## Cautions

- The assistant requires an OpenAI API key
- Large datasets may take longer to process
- Always verify AI-generated annotations with domain expertise
- The assistant is designed to assist, not replace, human expertise
- API usage incurs costs based on your OpenAI account