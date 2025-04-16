import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from scipy import stats
import umap
import scanpy as sc
import anndata as ad
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Set page config
st.set_page_config(
    page_title="Cell Type Annotation Assistant",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state for storing data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'anndata' not in st.session_state:
    st.session_state.anndata = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {}

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

VISUALIZATION_PROMPT = """You are an expert in single-cell RNA sequencing visualization and interpretation, following the methodology from the VisCello paper.
Your task is to help users interpret visualization of their single-cell data.

Current data context:
{data_context}

Visualization context:
{viz_context}

User question: {question}

Please provide a detailed interpretation that includes:
1. Analysis of the clustering patterns visible in the visualization
2. Identification of potential batch effects or technical artifacts
3. Suggestions for alternative visualization approaches
4. Biological implications of the observed patterns
5. Recommendations for further analysis
"""

CLUSTERING_PROMPT = """You are an expert in single-cell RNA sequencing clustering analysis.
Your task is to help users interpret and optimize the clustering of their single-cell data.

Current data context:
{data_context}

Clustering context:
{cluster_context}

User question: {question}

Please provide a detailed analysis that includes:
1. Assessment of the current clustering quality
2. Recommendations for parameter adjustments to improve clustering
3. Biological interpretation of the identified clusters
4. Suggestions for marker genes that define each cluster
5. Possible sub-clustering opportunities for heterogeneous clusters
"""

def setup_openai():
    """Set up OpenAI API with user's API key"""
    api_key = st.session_state.api_key
    return OpenAI(temperature=0, openai_api_key=api_key)

def create_prompt_chain(template_text):
    """Create a LangChain prompt template and chain"""
    template = PromptTemplate(
        input_variables=["data_context", "question"],
        template=template_text
    )
    return LLMChain(llm=setup_openai(), prompt=template)

def create_viz_prompt_chain():
    """Create a visualization-specific prompt chain"""
    template = PromptTemplate(
        input_variables=["data_context", "viz_context", "question"],
        template=VISUALIZATION_PROMPT
    )
    return LLMChain(llm=setup_openai(), prompt=template)

def create_clustering_prompt_chain():
    """Create a clustering-specific prompt chain"""
    template = PromptTemplate(
        input_variables=["data_context", "cluster_context", "question"],
        template=CLUSTERING_PROMPT
    )
    return LLMChain(llm=setup_openai(), prompt=template)

def calculate_differential_genes(df, cluster1, cluster2):
    """Calculate differentially expressed genes between two clusters using Wilcoxon test."""
    genes = df['gene'].values
    expr1 = df[cluster1].values
    expr2 = df[cluster2].values
    
    p_values = []
    fold_changes = []
    for i in range(len(genes)):
        _, p_val = stats.ranksums(expr1[i], expr2[i])
        p_values.append(p_val)
        
        # Calculate log2 fold change
        eps = 1e-10  # Small value to avoid division by zero
        log2fc = np.log2((np.mean(expr1[i]) + eps) / (np.mean(expr2[i]) + eps))
        fold_changes.append(log2fc)
    
    # Create a DataFrame for sorting and filtering
    diff_df = pd.DataFrame({
        'gene': genes,
        'p_value': p_values,
        'log2fc': fold_changes
    })
    
    # Sort by p-value
    diff_df = diff_df.sort_values('p_value')
    
    # Return top 10 genes and their stats
    return diff_df.head(10)

def visualize_gene_expression(df, gene, plot_type="violin"):
    """Create visualizations for gene expression across clusters."""
    if plot_type == "violin":
        # Melt the DataFrame for plotting
        plot_df = pd.melt(df[['gene'] + [col for col in df.columns if col != 'gene']], 
                          id_vars=['gene'], 
                          var_name='cluster', 
                          value_name='expression')
        plot_df = plot_df[plot_df['gene'] == gene]
        
        fig = px.violin(plot_df, x='cluster', y='expression', box=True, points="all")
        fig.update_layout(title=f"Expression of {gene} across clusters")
        return fig
    
    elif plot_type == "heatmap":
        # Get top genes by variance
        gene_vars = df.set_index('gene').var(axis=1).sort_values(ascending=False).head(15).index
        heatmap_df = df[df['gene'].isin(gene_vars)].set_index('gene')
        
        fig = px.imshow(heatmap_df, 
                        labels=dict(x="Cluster", y="Gene", color="Expression"),
                        color_continuous_scale='viridis')
        fig.update_layout(title="Gene Expression Heatmap")
        return fig
    
    elif plot_type == "bar":
        # For a specific gene across clusters
        gene_data = df[df['gene'] == gene].melt(id_vars=['gene'], value_name='expression', var_name='cluster')
        fig = px.bar(gene_data, x='cluster', y='expression', color='cluster',
                     title=f"Expression of {gene} across clusters")
        return fig

def create_umap_plot(df):
    """Create a UMAP embedding for visualization."""
    # Prepare data matrix
    genes = df['gene'].values
    data_matrix = df.drop('gene', axis=1).values
    
    # Run UMAP
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(data_matrix)
    
    # Create a DataFrame for the embedding
    umap_df = pd.DataFrame({
        'gene': genes,
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1]
    })
    
    # Create a scatter plot
    fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', hover_name='gene',
                     title='UMAP Visualization of Gene Expression')
    
    return fig

def convert_to_anndata(df):
    """Convert the DataFrame to AnnData object for scanpy analysis."""
    # Extract gene names and expression data
    genes = df['gene'].values
    data_matrix = df.drop('gene', axis=1).values
    cluster_names = [col for col in df.columns if col != 'gene']
    
    # Create AnnData object
    adata = ad.AnnData(X=data_matrix.T)  # Note: AnnData expects genes as columns, cells as rows
    adata.var_names = genes  # Genes as features
    adata.obs_names = cluster_names  # Clusters as observations
    
    return adata

def analyze_with_scanpy(adata):
    """Perform basic scanpy analysis on the data."""
    # Normalize data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Principal component analysis
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Run UMAP
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata)
    
    return adata

def evaluate_predictions(predictions, ground_truth):
    """Evaluate predictions against ground truth."""
    # Check if we have matching IDs
    if len(predictions) != len(ground_truth):
        st.warning("Prediction and ground truth counts don't match.")
        return None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
        'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
        'f1_score': f1_score(ground_truth, predictions, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(ground_truth, predictions)
    }
    
    return metrics

def log_evaluation(metrics, query, prediction_details):
    """Log evaluation metrics and details to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        'timestamp': timestamp,
        'metrics': metrics,
        'query': query,
        'predictions': prediction_details
    }
    
    # Create directory if it doesn't exist
    os.makedirs('evaluation/logs', exist_ok=True)
    
    # Append to log file
    log_file = 'evaluation/logs/prediction_log.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Update session state
    st.session_state.evaluation_metrics = metrics
    st.session_state.prediction_history.append(log_entry)
    
    return log_file

def main():
    st.title("🧬 Cell Type Annotation Assistant")
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key", type="password", 
                               help="Your API key will be used to interact with the OpenAI API.")
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key set successfully!")
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        page = st.radio("Select Page", ["Data Analysis", "Visualization", "Clustering", "Evaluation"])
        
        st.divider()
        
        # Metrics display if available
        if st.session_state.evaluation_metrics:
            st.header("Latest Evaluation Metrics")
            for metric, value in st.session_state.evaluation_metrics.items():
                st.metric(label=metric.capitalize(), value=f"{value:.4f}")
    
    # Check if API key is set
    if not st.session_state.api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
        return
    
    # File upload section (common across pages)
    if st.session_state.data is None:
        st.subheader("Upload your single-cell data")
        
        uploaded_file = st.file_uploader("Upload your single-cell data (CSV format)", type=['csv'])
        
        if uploaded_file is not None:
            # Load data
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Try to convert to AnnData
                try:
                    st.session_state.anndata = convert_to_anndata(df)
                    st.success("Data loaded and converted to AnnData format successfully!")
                except Exception as e:
                    st.warning(f"Could not convert to AnnData: {str(e)}")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    else:
        # Display current data
        with st.expander("Data Preview", expanded=False):
            st.dataframe(st.session_state.data.head())
        
        # Option to upload new data
        if st.button("Upload New Data"):
            st.session_state.data = None
            st.session_state.anndata = None
            st.session_state.prediction_history = []
            st.experimental_rerun()
    
    # Only proceed if data is loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Create data context
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
                diff_genes_df = calculate_differential_genes(df, clusters[i], clusters[j])
                data_context += f"\n{clusters[i]} vs {clusters[j]}:\n"
                for _, row in diff_genes_df.iterrows():
                    data_context += f"- {row['gene']}: p-value={row['p_value']:.6f}, log2FC={row['log2fc']:.4f}\n"
        
        # Different content based on selected page
        if page == "Data Analysis":
            st.header("Cell Type Analysis")
            
            # More detailed query options
            query_type = st.selectbox("Select Query Type", [
                "Cell Type Identification",
                "Marker Gene Analysis",
                "Differential Expression Interpretation",
                "Custom Query"
            ])
            
            # Predefined questions based on query type
            if query_type == "Cell Type Identification":
                question_template = "What cell types are likely present in this dataset based on the marker genes?"
            elif query_type == "Marker Gene Analysis":
                question_template = "What do the marker genes in cluster {cluster} suggest about its cell identity?"
                # Let user select a cluster
                selected_cluster = st.selectbox("Select Cluster", clusters)
                question_template = question_template.format(cluster=selected_cluster)
            elif query_type == "Differential Expression Interpretation":
                question_template = "Interpret the differential expression between clusters {cluster1} and {cluster2}."
                # Let user select clusters
                col1, col2 = st.columns(2)
                with col1:
                    selected_cluster1 = st.selectbox("Select Cluster 1", clusters)
                with col2:
                    remaining_clusters = [c for c in clusters if c != selected_cluster1]
                    selected_cluster2 = st.selectbox("Select Cluster 2", remaining_clusters)
                question_template = question_template.format(cluster1=selected_cluster1, cluster2=selected_cluster2)
            else:  # Custom Query
                question_template = ""
            
            # User input
            user_question = st.text_area(
                "Ask a detailed question about your data:",
                value=question_template,
                height=100
            )
            
            # Process the query
            if st.button("Analyze"):
                if user_question:
                    with st.spinner("Analyzing your data using GPT..."):
                        try:
                            # Create chain with the current API key
                            chain = create_prompt_chain(CELL_TYPE_PROMPT)
                            
                            # Get response from LLM
                            response = chain.run(data_context=data_context, question=user_question)
                            
                            # Display response
                            st.subheader("Analysis")
                            st.write(response)
                            
                            # Add visualization
                            st.subheader("Gene Expression Visualization")
                            viz_options = st.multiselect(
                                "Select visualization options:",
                                ["Violin Plot", "Heatmap", "Bar Chart"]
                            )
                            
                            if "Violin Plot" in viz_options:
                                selected_gene = st.selectbox("Select a gene for violin plot:", df['gene'].values, key="violin")
                                fig = visualize_gene_expression(df, selected_gene, plot_type="violin")
                                st.plotly_chart(fig)
                            
                            if "Heatmap" in viz_options:
                                fig = visualize_gene_expression(df, None, plot_type="heatmap")
                                st.plotly_chart(fig)
                            
                            if "Bar Chart" in viz_options:
                                selected_gene = st.selectbox("Select a gene for bar chart:", df['gene'].values, key="bar")
                                fig = visualize_gene_expression(df, selected_gene, plot_type="bar")
                                st.plotly_chart(fig)
                            
                            # Log this query to history
                            st.session_state.prediction_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'query': user_question,
                                'response': response
                            })
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                else:
                    st.warning("Please enter a question to analyze.")
        
        elif page == "Visualization":
            st.header("Data Visualization")
            
            # UMAP visualization
            try:
                if st.button("Generate UMAP Visualization"):
                    with st.spinner("Generating UMAP plot..."):
                        umap_fig = create_umap_plot(df)
                        st.plotly_chart(umap_fig)
                
                # More VisCello-like visualizations
                st.subheader("Advanced Visualizations")
                
                viz_type = st.selectbox("Select Visualization Type", [
                    "Gene Expression Heatmap",
                    "Cluster Comparison",
                    "Marker Gene Violin Plots"
                ])
                
                if viz_type == "Gene Expression Heatmap":
                    fig = visualize_gene_expression(df, None, plot_type="heatmap")
                    st.plotly_chart(fig)
                
                elif viz_type == "Cluster Comparison":
                    col1, col2 = st.columns(2)
                    with col1:
                        cluster1 = st.selectbox("Select Cluster 1", clusters, key="cluster1_viz")
                    with col2:
                        cluster2 = st.selectbox("Select Cluster 2", clusters, key="cluster2_viz")
                    
                    # Show differential expression
                    if cluster1 != cluster2:
                        diff_genes = calculate_differential_genes(df, cluster1, cluster2)
                        st.write(f"Top differentially expressed genes between {cluster1} and {cluster2}:")
                        st.dataframe(diff_genes)
                        
                        # Generate plots for top genes
                        for _, row in diff_genes.head(3).iterrows():
                            gene = row['gene']
                            fig = visualize_gene_expression(df, gene, plot_type="violin")
                            st.plotly_chart(fig)
                
                elif viz_type == "Marker Gene Violin Plots":
                    # Allow multiple gene selection
                    selected_genes = st.multiselect("Select marker genes to visualize:", df['gene'].values)
                    
                    if selected_genes:
                        for gene in selected_genes:
                            fig = visualize_gene_expression(df, gene, plot_type="violin")
                            st.plotly_chart(fig)
                
                # Interpret visualization with AI
                st.subheader("Interpret Visualizations")
                viz_context = f"Visualization type: {viz_type}"
                viz_question = st.text_area(
                    "Ask a question about the visualization:",
                    placeholder="e.g., What patterns do you see in the gene expression heatmap?",
                    key="viz_question"
                )
                
                if st.button("Interpret") and viz_question:
                    with st.spinner("Generating interpretation..."):
                        try:
                            # Create chain for visualization interpretation
                            viz_chain = create_viz_prompt_chain()
                            
                            # Get response
                            viz_response = viz_chain.run(
                                data_context=data_context,
                                viz_context=viz_context,
                                question=viz_question
                            )
                            
                            st.write("### Interpretation")
                            st.write(viz_response)
                            
                        except Exception as e:
                            st.error(f"Error generating interpretation: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error in visualization: {str(e)}")
        
        elif page == "Clustering":
            st.header("Clustering Analysis")
            
            # Only proceed if we have AnnData
            if st.session_state.anndata is not None:
                adata = st.session_state.anndata
                
                # Run scanpy analysis
                if st.button("Run Clustering Analysis"):
                    with st.spinner("Running analysis with scanpy..."):
                        try:
                            # Analyze with scanpy
                            adata_analyzed = analyze_with_scanpy(adata)
                            
                            # Store results
                            st.session_state.anndata = adata_analyzed
                            
                            st.success("Analysis complete!")
                            
                            # Show UMAP plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sc.pl.umap(adata_analyzed, color=['leiden'], ax=ax, show=False)
                            st.pyplot(fig)
                            
                            # Show cluster information
                            st.subheader("Cluster Information")
                            cluster_counts = adata_analyzed.obs['leiden'].value_counts()
                            st.write(f"Number of clusters: {len(cluster_counts)}")
                            st.dataframe(pd.DataFrame({
                                'Cluster': cluster_counts.index,
                                'Cell Count': cluster_counts.values
                            }))
                            
                            # Prepare context for AI interpretation
                            cluster_context = f"""
                            Number of clusters identified: {len(cluster_counts)}
                            Cluster sizes: {dict(cluster_counts)}
                            Clustering method: Leiden algorithm
                            Resolution parameter: default
                            """
                            
                            # AI interpretation of clusters
                            st.subheader("Interpret Clustering")
                            cluster_question = st.text_area(
                                "Ask a question about the clustering results:",
                                placeholder="e.g., What do the clustering results suggest about cell populations?",
                                key="cluster_question"
                            )
                            
                            if cluster_question:
                                with st.spinner("Interpreting clusters..."):
                                    # Create chain for clustering interpretation
                                    cluster_chain = create_clustering_prompt_chain()
                                    
                                    # Get response
                                    cluster_response = cluster_chain.run(
                                        data_context=data_context,
                                        cluster_context=cluster_context,
                                        question=cluster_question
                                    )
                                    
                                    st.write("### Cluster Interpretation")
                                    st.write(cluster_response)
                            
                        except Exception as e:
                            st.error(f"Error in clustering analysis: {str(e)}")
                
            else:
                st.warning("Could not convert data to AnnData format for clustering analysis.")
                if st.button("Try to convert data again"):
                    try:
                        st.session_state.anndata = convert_to_anndata(df)
                        st.success("Successfully converted data to AnnData format!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Conversion error: {str(e)}")
        
        elif page == "Evaluation":
            st.header("Evaluation & Metrics")
            
            # Upload ground truth data
            st.subheader("Upload Ground Truth Data")
            ground_truth_file = st.file_uploader("Upload ground truth labels (CSV)", type=['csv'])
            
            if ground_truth_file is not None:
                # Load ground truth
                ground_truth_df = pd.read_csv(ground_truth_file)
                
                st.write("Ground Truth Preview:")
                st.dataframe(ground_truth_df.head())
                
                # Run evaluation
                st.subheader("Run Evaluation")
                
                # Simple cell type prediction
                st.write("Cell Type Prediction for Evaluation:")
                
                # Select which cluster to evaluate
                cluster_to_evaluate = st.selectbox("Select cluster to evaluate:", clusters)
                
                if st.button("Run Prediction and Evaluation"):
                    with st.spinner("Running evaluation..."):
                        try:
                            # Create chain with the current API key
                            chain = create_prompt_chain(CELL_TYPE_PROMPT)
                            
                            # Prepare focused data context
                            eval_data_context = f"""
                            Evaluating cluster: {cluster_to_evaluate}
                            
                            Top marker genes for {cluster_to_evaluate}:
                            """
                            
                            # Add marker genes
                            for other_cluster in clusters:
                                if other_cluster != cluster_to_evaluate:
                                    diff_genes = calculate_differential_genes(df, cluster_to_evaluate, other_cluster)
                                    top_genes = diff_genes['gene'].values[:5]
                                    eval_data_context += f"\nvs {other_cluster}: {', '.join(top_genes)}"
                            
                            # Run prediction
                            prediction_query = f"What cell type is most likely represented by cluster {cluster_to_evaluate}? Provide a single cell type name."
                            prediction_response = chain.run(data_context=eval_data_context, question=prediction_query)
                            
                            # Display prediction
                            st.write("### Prediction Result")
                            st.write(prediction_response)
                            
                            # Extract cell type from response (assuming it's the first line or a clear statement)
                            import re
                            cell_type_match = re.search(r"(T-cell|B-cell|Macrophage|Neutrophil|NK cell|Monocyte|Dendritic cell)", 
                                                        prediction_response)
                            
                            if cell_type_match:
                                predicted_cell_type = cell_type_match.group(1)
                                
                                # Check ground truth
                                true_cell_type = ground_truth_df['cell_type'].iloc[0]  # Just using the first one for demo
                                
                                # Calculate metrics
                                metrics = {
                                    'accuracy': 1.0 if predicted_cell_type == true_cell_type else 0.0,
                                    'precision': 1.0 if predicted_cell_type == true_cell_type else 0.0,
                                    'recall': 1.0 if predicted_cell_type == true_cell_type else 0.0,
                                    'f1_score': 1.0 if predicted_cell_type == true_cell_type else 0.0,
                                    'cohen_kappa': 1.0 if predicted_cell_type == true_cell_type else 0.0
                                }
                                
                                # Log evaluation
                                prediction_details = [{
                                    'cluster': cluster_to_evaluate,
                                    'predicted': predicted_cell_type,
                                    'true': true_cell_type,
                                    'correct': predicted_cell_type == true_cell_type
                                }]
                                
                                log_file = log_evaluation(metrics, prediction_query, prediction_details)
                                
                                # Display evaluation results
                                st.write("### Evaluation Results")
                                st.write(f"Predicted: {predicted_cell_type}")
                                st.write(f"True: {true_cell_type}")
                                st.write(f"Correct: {predicted_cell_type == true_cell_type}")
                                
                                # Display metrics
                                st.write("### Metrics")
                                for metric, value in metrics.items():
                                    st.metric(label=metric.capitalize(), value=f"{value:.4f}")
                                
                                st.success(f"Evaluation logged to {log_file}")
                            else:
                                st.warning("Could not extract a clear cell type prediction from the response.")
                            
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                
                # Show prediction history
                if st.session_state.prediction_history:
                    st.subheader("Prediction History")
                    for i, entry in enumerate(st.session_state.prediction_history):
                        with st.expander(f"Prediction {i+1} - {entry.get('timestamp', 'Unknown time')}"):
                            st.write(f"**Query:** {entry.get('query', 'N/A')}")
                            if 'metrics' in entry:
                                st.write("**Metrics:**")
                                for metric, value in entry['metrics'].items():
                                    st.write(f"- {metric.capitalize()}: {value:.4f}")
                            if 'predictions' in entry:
                                st.write("**Predictions:**")
                                for pred in entry['predictions']:
                                    st.write(f"- Cluster: {pred.get('cluster', 'N/A')}")
                                    st.write(f"  Predicted: {pred.get('predicted', 'N/A')}")
                                    st.write(f"  True: {pred.get('true', 'N/A')}")
                                    st.write(f"  Correct: {pred.get('correct', False)}")
            else:
                st.info("Please upload ground truth data to perform evaluation.")
    
    # Footer
    st.divider()
    st.write("Cell Type Annotation Assistant - Based on GPTCelltype and VisCello methodologies")

if __name__ == "__main__":
    main()