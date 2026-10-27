# Porter Delivery Time Estimator - Gradio App

A Gradio-based web interface for the Porter delivery time estimation neural network model.

## Features
- Single prediction form with all relevant input features
- Real-time ETA prediction in minutes
- Model performance metrics display
- Deployable to Hugging Face Spaces

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Train and save model artifacts (run once)
python train_and_save_model.py

# Run the Gradio app
python app.py
```

### Deploy to Hugging Face Spaces
1. Create a new Space on Hugging Face (select "Gradio" as SDK)
2. Push this repository to the Space
3. The app will auto-deploy

## Project Structure
```
porter_ui/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── train_and_save_model.py # Training script (run once)
├── model_artifacts/       # Saved model, scaler, features (generated)
│   ├── delivery_time_model.keras
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── feature_columns.json
│   ├── categorical_info.json
│   └── metrics.json
└── README.md
```

## Input Features
The model expects the following features (matching the trained model):
- **Market**: market_id
- **Order details**: total_items, subtotal, num_distinct_items, min_item_price, max_item_price
- **Delivery partners (Dashers)**: total_onshift_dashers, total_busy_dashers, total_outstanding_orders
- **Distance**: estimated_store_to_consumer_driving_duration (minutes)

## Model Performance (Actual)
- **MAE (Mean Absolute Error):** 3.21 minutes
- **RMSE (Root Mean Squared Error):** 4.36 minutes
- **R² Score:** 0.783
- **Training Samples:** 140,621
- **Test Samples:** 35,156
- **Features Used:** 10

## License
MIT
