#!/bin/bash
# Startup script for Porter Delivery Time Estimator

echo "🚚 Starting Porter Delivery Time Estimator..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if model artifacts exist
if [ ! -f "model_artifacts/delivery_time_model.keras" ]; then
    echo "🤖 Model artifacts not found. Training model..."
    cd ..
    python train_and_save_model.py
    cp -r model_artifacts/* porter_ui/model_artifacts/
    cd porter_ui
fi

# Start Gradio app
echo "🌐 Starting Gradio server on http://localhost:7860"
echo "Press Ctrl+C to stop"
python app.py