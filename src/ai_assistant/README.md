# Cell Type Annotation AI Assistant

This is an AI-powered assistant designed to help with cell type annotation in single-cell RNA sequencing analysis. It's part of the larger GPTCelltype enhancement project.

## Features

- Interactive data upload and analysis
- Natural language queries about cell types and marker genes
- AI-powered analysis and recommendations
- Visualization of gene expression patterns
- Integration with GPTCelltype workflow

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

3. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your single-cell data (CSV format)
3. Ask questions about your data, such as:
   - "What cell types might be present in this dataset?"
   - "Which genes are potential markers for cluster X?"
   - "How can I improve the cell type annotation?"

## Cautions

- The assistant requires an OpenAI API key
- Large datasets may take longer to process
- Always verify AI-generated annotations with domain expertise
- The assistant is designed to assist, not replace, human expertise

## Integration with GPTCelltype

This assistant is designed to complement the GPTCelltype package by:
1. Providing preliminary analysis before using GPTCelltype
2. Helping interpret GPTCelltype results
3. Suggesting improvements to annotation workflows

## Evaluation

The assistant's performance is evaluated based on:
- Accuracy of cell type predictions
- Relevance of marker gene suggestions
- Quality of biological context provided
- User feedback and interaction quality 