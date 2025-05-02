# Cell Type Annotation AI Assistant

This is an AI-powered assistant designed to help with cell type annotation in single-cell RNA sequencing analysis. It's part of the larger GPTCelltype enhancement project with integrations for VisCello-like visualizations.

## Features

- Interactive data upload and analysis
- Support for multiple single-cell RNA-seq file formats (AnnData, H5AD, Loom, 10X, CSV)
- Retrieval Augmented Generation (RAG) with domain-specific knowledge
- User-provided context for customized analysis
- Natural language queries about cell types and marker genes
- AI-powered analysis and recommendations with GPT
- Advanced visualizations including UMAP, heatmaps, and violin plots
- Integration with GPTCelltype and VisCello workflows
- Scanpy-based clustering analysis
- Performance evaluation with accuracy, precision, recall, F1, and Cohen's kappa
- Testing and validation suite

## Setup and Running

### Quick Start

The easiest way to run the application is using the launch script:

```bash
./launch.sh
```

This will:
1. Create a virtual environment if needed
2. Install all required dependencies
3. Set up necessary directories
4. Launch the application

### Compatibility Mode

If you encounter segmentation faults or other crashes (especially on macOS with Python 3.12), use compatibility mode:

```bash
./launch.sh --compatible
```

Compatibility mode:
- Skips problematic dependencies like FAISS (uses SimpleFallbackRetriever instead)
- Uses more conservative threading settings to prevent segmentation faults
- Disables multi-threading in NumPy and OpenMP that often cause crashes
- Pins specific package versions known to work well (zarr, numcodecs, etc.)
- Sets environment variables to prevent thread-related crashes
- Creates fallback mechanisms for all major features
- Provides more stable operation on challenging systems

### Manual Setup

If you prefer manual setup:

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
2. Upload your single-cell data in any supported format:
   - H5AD (AnnData)
   - CSV (with gene names in the first column)
   - Loom files
   - 10X HDF5 files
3. Choose from multiple analysis options:
   - **Data Analysis**: Identify cell types, explore marker genes, interpret differential expression
   - **Visualization**: Generate UMAP plots, heatmaps, and compare clusters
   - **Clustering**: Perform clustering analysis with scanpy
   - **Evaluation**: Compare predictions against ground truth data
   - **RAG Context**: Add and query domain-specific knowledge
   - **Testing**: Run validation tests for application functionality

### Analysis Features

- Query templates for common analysis tasks
- Interactive visualizations for gene expression patterns
- AI-powered interpretation of visualizations
- Clustering and dimensional reduction
- Evaluation metrics tracking and reporting

## RAG Knowledge Base

The application includes a Retrieval Augmented Generation (RAG) system that enhances LLM responses with relevant domain knowledge:

- **PanglaoDB Integration**: Dynamically accesses cell marker information from PanglaoDB database for accurate cell type identification
- **Baseline Knowledge**: Pre-loaded information about scRNA-seq analysis, cell types, and marker genes
- **User Context**: Upload your own research papers or domain-specific text to customize the assistant
- **Knowledge Retrieval**: The system automatically retrieves relevant context based on your queries
- **Compatibility Mode**: Falls back to simpler retrieval when FAISS isn't available

For cell marker and cell type queries, the system will:
1. Detect if you're asking about specific cell types or markers
2. Connect to PanglaoDB's knowledge base to retrieve up-to-date marker information
3. Combine this with the built-in knowledge and your custom context
4. Return comprehensive, accurate responses with specificity scores when available

### FAISS and Compatibility Mode

The RAG system works optimally with FAISS (Facebook AI Similarity Search) for efficient vector search. If FAISS isn't available:

1. The app automatically switches to a simplified retrieval mechanism
2. A **SimpleFallbackRetriever** provides basic keyword matching
3. PanglaoDB integration continues to work for marker gene queries
4. All core functionality remains available, with slightly reduced retrieval precision

You can install FAISS for optimal performance:
```bash
pip install faiss-cpu
```

For ARM-based systems (M1/M2 Macs), FAISS may require special installation via conda:
```bash
conda install -c conda-forge faiss
```

### Adding Your Own Context

When you add custom context, it's applied to **ALL** analysis functions throughout the application:

1. Navigate to the "RAG Context" page
2. Upload publications or text files (PDF, TXT, CSV, MD formats supported)
3. The system will process the document and add it to the knowledge base
4. Your custom context will be used in all subsequent analyses across all pages, including:
   - Data Analysis: Cell type identification and marker gene interpretation
   - Visualization: Interpretation of plots and expression patterns
   - Clustering: Cluster analysis and cell population identification
   - Evaluation: Cell type predictions against ground truth

This allows you to provide domain-specific information (like a research paper relevant to your dataset) that will inform all analyses performed within the app.

### Saving Analysis Results

The app provides options to save your analysis results:
- Download analysis reports as markdown files 
- Results include timestamps, queries, and complete analysis text
- Cell type predictions can be saved for future reference

## Testing and Validation

The application includes a testing and validation framework:

- **Validation Suite**: Runs tests on core functionality
- **Validation Log**: Tracks test results for troubleshooting
- **Format Support**: Validates multiple file format loading
- **RAG Testing**: Tests knowledge base retrieval
- **Evaluation Tracking**: Measures prediction accuracy

To run tests:
1. Navigate to the "Testing" page
2. Click "Run Tests" to execute the validation suite
3. View test results and validation history

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
6. Adding RAG capabilities for more context-aware analysis

## Cautions

- The assistant requires an OpenAI API key
- Large datasets may take longer to process
- Always verify AI-generated annotations with domain expertise
- The assistant is designed to assist, not replace, human expertise
- API usage incurs costs based on your OpenAI account
- For MTX format support, please use the H5AD conversion option in the scanpy package