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

# Create evaluation directories if they don't exist
echo "Setting up evaluation directories..."
mkdir -p evaluation/logs evaluation/reports

# Check for .env file but don't create it with a key - user will enter their key in the app
if [ ! -f ".env" ]; then
    echo "Creating empty .env file..."
    echo "# Add your OpenAI API key through the app interface" > .env
    echo "# Format: OPENAI_API_KEY=your_key_here" >> .env
fi

# Launch the Streamlit app
echo "Launching Cell Type Annotation Assistant..."
streamlit run app.py