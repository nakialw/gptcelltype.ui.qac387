import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import seaborn as sns
from dotenv import load_dotenv
import os
from scipy import stats

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Cell Type Annotation Assistant",
    page_icon="🧬",
    layout="wide"
)

# Initialize OpenAI
llm = OpenAI(temperature=0)

# Define the system prompt based on GPTCelltype paper methodology
CELL_TYPE_PROMPT = """You are an expert in single-cell RNA sequencing analysis and cell type annotation, following the methodology from the GPTCelltype paper.
Your task is to help users analyze their single-cell data and identify cell types based on marker genes.

Current data context:
{data_context}

User question: {question}

Please provide a detailed response that includes:
1. Analysis of the provided data using the top differentially expressed genes
2. Recommendations for cell type identification based on marker genes
3. Biological context and tissue-specific considerations
4. Confidence levels and potential caveats
5. Comparison with known cell type markers from literature

Note: For optimal results, use the top 10 differentially expressed genes derived from the Wilcoxon test, as shown in the GPTCelltype paper.
"""

# Create prompt template
prompt_template = PromptTemplate(
    input_variables=["data_context", "question"],
    template=CELL_TYPE_PROMPT
)

# Create LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)

def calculate_differential_genes(df, cluster1, cluster2):
    """Calculate differentially expressed genes between two clusters using Wilcoxon test."""
    genes = df['gene'].values
    expr1 = df[cluster1].values
    expr2 = df[cluster2].values
    
    p_values = []
    for i in range(len(genes)):
        _, p_val = stats.ranksums(expr1[i], expr2[i])
        p_values.append(p_val)
    
    # Sort by p-value and return top 10 genes
    sorted_indices = np.argsort(p_values)[:10]
    return genes[sorted_indices]

def visualize_gene_expression(df, gene):
    """Create a violin plot for gene expression across clusters."""
    fig = px.violin(df, y=gene, box=True, points="all")
    fig.update_layout(title=f"Expression of {gene} across clusters")
    return fig

def main():
    st.title("🧬 Cell Type Annotation Assistant")
    st.write("""
    This AI assistant helps you analyze single-cell RNA sequencing data and identify cell types.
    Based on the GPTCelltype paper methodology, it uses GPT-4 to analyze marker genes and provide cell type annotations.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload your single-cell data (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Create data context with differential expression analysis
        clusters = [col for col in df.columns if col != 'gene']
        data_context = f"""
        Dataset shape: {df.shape}
        Clusters: {', '.join(clusters)}
        Number of genes: {len(df)}
        
        Top differentially expressed genes between clusters:
        """
        
        # Add differential expression analysis for each pair of clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                diff_genes = calculate_differential_genes(df, clusters[i], clusters[j])
                data_context += f"\n{clusters[i]} vs {clusters[j]}: {', '.join(diff_genes)}"
        
        # User input
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What cell types might be present in this dataset?",
            key="user_question"
        )
        
        if user_question:
            with st.spinner("Analyzing your data using GPT-4..."):
                # Get response from LLM
                response = chain.run(data_context=data_context, question=user_question)
                
                # Display response
                st.subheader("Analysis")
                st.write(response)
                
                # Add visualization if relevant
                if "marker" in user_question.lower() or "expression" in user_question.lower():
                    st.subheader("Gene Expression Visualization")
                    selected_gene = st.selectbox("Select a gene to visualize:", df['gene'].values)
                    fig = visualize_gene_expression(df, selected_gene)
                    st.plotly_chart(fig)
                    
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main() 