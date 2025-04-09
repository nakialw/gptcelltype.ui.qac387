#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if not already installed
echo "Checking and installing requirements..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=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" > .env
    echo "Please edit .env file with your OpenAI API key"
    exit 1
fi

# Launch the Streamlit app
echo "Launching Cell Type Annotation Assistant..."
streamlit run app.py 