#!/bin/bash

# Enhanced launch script for Cell Type Annotation Assistant
# Includes compatibility mode for systems prone to segmentation faults

# Process command line arguments
COMPATIBILITY_MODE=0
HELP=0

for arg in "$@"; do
  case $arg in
    --help|-h)
      HELP=1
      ;;
    --compatible|--compatibility)
      COMPATIBILITY_MODE=1
      ;;
  esac
done

# Display help if requested
if [ $HELP -eq 1 ]; then
  echo "Launch script for Cell Type Annotation Assistant"
  echo ""
  echo "Usage:"
  echo "  ./launch.sh [options]"
  echo ""
  echo "Options:"
  echo "  --compatible    Launch in compatibility mode (no FAISS, reduced threading)"
  echo "  --help, -h      Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./launch.sh                # Standard launch"
  echo "  ./launch.sh --compatible   # For systems with segmentation faults"
  exit 0
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements with appropriate settings
echo "Checking and installing requirements..."
if [ $COMPATIBILITY_MODE -eq 1 ]; then
    echo "Using compatibility mode - adjusting dependencies..."
    
    # Install core dependencies except problematic ones
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
    pip install openai langchain langchain-openai langchain-community python-dotenv
    pip install scanpy==1.9.3 anndata==0.8.0 umap-learn==0.5.3
    
    # Pin specific versions known to work with minimal compatibility issues
    pip install zarr==2.14.2 numcodecs==0.11.0
    
    # Set environment variables to avoid threading issues
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export OMP_MAX_ACTIVE_LEVELS=1  # Replace deprecated omp_set_nested
    
    # Set FAISS_DISABLE environment variable to signal compatibility mode to the app
    export FAISS_DISABLE=1
    
    echo "Compatibility mode enabled - skipping FAISS and using simple fallback retriever"
else
    # Standard installation
    pip install -r requirements.txt
    
    # Try to install FAISS if not already installed
    echo "Checking FAISS installation..."
    python -c "import faiss" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "FAISS not found, attempting to install..."
        # Try pip install first
        pip install faiss-cpu
        
        # If pip install fails, provide information
        if [ $? -ne 0 ]; then
            echo "FAISS installation failed. The app will run in partial compatibility mode."
            echo "For optimal performance, install FAISS manually with:"
            echo "  pip install faiss-cpu"
            echo "or for conda:"
            echo "  conda install -c conda-forge faiss-cpu"
            
            # We'll still run, but using the fallback retriever
            export FAISS_DISABLE=1
        fi
    else
        echo "FAISS is already installed."
    fi
    
    # Set minimal environment variables
    export OMP_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export OMP_MAX_ACTIVE_LEVELS=1  # Replace deprecated omp_set_nested
fi

# Create evaluation directories if they don't exist
echo "Setting up evaluation directories..."
mkdir -p evaluation/logs evaluation/reports

# Check for .env file but don't create it with a key - user will enter their key in the app
if [ ! -f ".env" ]; then
    echo "Creating empty .env file..."
    echo "# Add your OpenAI API key through the app interface" > .env
    echo "# Format: OPENAI_API_KEY=your_key_here" >> .env
fi

# Display app launch banner
echo ""
echo "========================================================"
echo "    Launching Cell Type Annotation Assistant"
if [ $COMPATIBILITY_MODE -eq 1 ]; then
    echo "    [COMPATIBILITY MODE]"
fi
echo "========================================================"
echo ""
echo "Note: Ignore these warnings (they don't affect functionality):"
echo "  - LangChain deprecation warnings"
echo "  - AI Message warnings"
echo "  - Resource tracker warnings"
echo "  - OMP nested routine deprecated warnings"
echo ""

# Launch the app with appropriate settings
if [ $COMPATIBILITY_MODE -eq 1 ]; then
    # Compatibility mode launch with maximum stability settings
    echo "Launching with compatibility settings for maximum stability..."
    PYTHONUNBUFFERED=1 STREAMLIT_SERVER_HEADLESS=1 python -m streamlit run app.py
else
    # Standard launch
    echo "Launching app in standard mode..."
    PYTHONUNBUFFERED=1 streamlit run app.py
    
    # If standard mode fails, try more conservative settings
    if [ $? -ne 0 ]; then
        echo ""
        echo "Standard launch failed. Trying with more conservative settings..."
        echo "If this also fails, try: ./launch.sh --compatible"
        echo ""
        # More conservative launch settings
        PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m streamlit run app.py
    fi
fi

# If everything failed, suggest compatibility mode
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: App failed to launch."
    echo ""
    echo "Please try launching in compatibility mode:"
    echo "  ./launch.sh --compatible"
    echo ""
fi