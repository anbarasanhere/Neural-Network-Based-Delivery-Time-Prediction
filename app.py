"""
Gradio UI for Porter Delivery Time Estimation
Run with: python app.py
"""
import gradio as gr
import pandas as pd
import numpy as np
import json
import pickle
import os

# Try to import tensorflow, with fallback warning
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using lightweight fallback")
    keras = None

def load_artifacts():
    """Load model, scaler, imputer, and feature columns"""
    try:
        model = keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'delivery_time_model.keras'))
        with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'imputer.pkl'), 'rb') as f:
            imputer = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.json'), 'r') as f:
            feature_columns = json.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'categorical_info.json'), 'r') as f:
            categorical_info = json.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        return model, scaler, imputer, feature_columns, categorical_info, metrics
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None, None, None, None

model, scaler, imputer, feature_columns, categorical_info, metrics = load_artifacts()

# ============ PREDICTION FUNCTION ============
def predict_delivery_time(
    # Order details (matching actual feature columns)
    market_id: int,
    total_items: int,
    subtotal: float,
    num_distinct_items: int,
    min_item_price: float,
    max_item_price: float,
    # Delivery partners
    total_onshift_dashers: int,
    total_busy_dashers: int,
    total_outstanding_orders: int,
    # Distance/travel
    estimated_store_to_consumer_driving_duration: float
):
    """Predict delivery time in minutes"""
    
    if model is None:
        return "❌ Model not loaded. Please run train_and_save_model.py first."
    
    try:
        # Create input dataframe with all features initialized to 0
        input_data = {col: 0 for col in feature_columns}
        
        # Set numeric features (matching actual feature columns)
        input_data['market_id'] = market_id
        input_data['total_items'] = total_items
        input_data['subtotal'] = subtotal
        input_data['num_distinct_items'] = num_distinct_items
        input_data['min_item_price'] = min_item_price
        input_data['max_item_price'] = max_item_price
        input_data['total_onshift_dashers'] = total_onshift_dashers
        input_data['total_busy_dashers'] = total_busy_dashers
        input_data['total_outstanding_orders'] = total_outstanding_orders
        input_data['estimated_store_to_consumer_driving_duration'] = estimated_store_to_consumer_driving_duration
        
        # Convert to DataFrame
        X_input = pd.DataFrame([input_data])
        
        # Ensure column order matches training
        X_input = X_input[feature_columns]
        
        # Impute missing values
        X_imputed = imputer.transform(X_input)
        X_imputed = pd.DataFrame(X_imputed, columns=feature_columns)
        
        # Scale features
        X_scaled = scaler.transform(X_imputed)
        
        # Predict (log scale)
        y_pred_log = model.predict(X_scaled, verbose=0).flatten()[0]
        
        # Convert back to minutes
        y_pred_minutes = np.expm1(y_pred_log)
        
        # Format output
        result = f"🕐 **Estimated Delivery Time: {y_pred_minutes:.1f} minutes**"
        
        # Add confidence context based on model metrics
        mae = metrics.get('test_mae_minutes', 15)
        result += f"\n\n📊 *Model MAE: ±{mae:.1f} minutes (typical error range)*"
        
        if y_pred_minutes < 20:
            result += "\n🚀 **Fast delivery expected!**"
        elif y_pred_minutes < 40:
            result += "\n📦 **Standard delivery time**"
        else:
            result += "\n⏳ **Longer delivery - high demand or distance**"
            
        return result
        
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# ============ GRADIO INTERFACE ============
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Porter Delivery Time Estimator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚚 Porter Delivery Time Estimator
        **Neural Network powered ETA prediction for intra-city deliveries**
        
        Enter order details below to get an estimated delivery time.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🏪 Market & Order Details")
                market_id = gr.Number(label="Market ID", value=1, minimum=1, step=1)
                total_items = gr.Number(label="Total Items", value=3, minimum=1, step=1)
                subtotal = gr.Number(label="Subtotal (₹)", value=250, minimum=0, step=10)
                num_distinct_items = gr.Number(label="Distinct Items", value=2, minimum=1, step=1)
                min_item_price = gr.Number(label="Min Item Price (₹)", value=50, minimum=0, step=5)
                max_item_price = gr.Number(label="Max Item Price (₹)", value=150, minimum=0, step=5)
                
                gr.Markdown("### 🚴 Delivery Partners (Dashers)")
                total_onshift_dashers = gr.Number(label="On-shift Dashers", value=15, minimum=0, step=1)
                total_busy_dashers = gr.Number(label="Busy Dashers", value=5, minimum=0, step=1)
                total_outstanding_orders = gr.Number(label="Outstanding Orders", value=8, minimum=0, step=1)
                
                gr.Markdown("### 📍 Distance")
                estimated_store_to_consumer_driving_duration = gr.Number(
                    label="Estimated Driving Duration (minutes)", 
                    value=15, 
                    minimum=0, 
                    step=1
                )
                
                predict_btn = gr.Button("🔮 Predict Delivery Time", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Prediction Result")
                output = gr.Markdown(label="Prediction")
                
                gr.Markdown("### 📈 Model Performance")
                if metrics:
                    gr.Markdown(f"""
                    - **MAE (Mean Absolute Error):** {metrics.get('test_mae_minutes', 'N/A'):.1f} minutes
                    - **RMSE (Root Mean Squared Error):** {metrics.get('test_rmse_minutes', 'N/A'):.1f} minutes
                    - **R² Score:** {metrics.get('test_r2', 'N/A'):.3f}
                    - **Training Samples:** {metrics.get('n_train_samples', 'N/A'):,}
                    - **Test Samples:** {metrics.get('n_test_samples', 'N/A'):,}
                    - **Features Used:** {metrics.get('n_features', 'N/A')}
                    """)
                else:
                    gr.Markdown("*Metrics will appear after model training*")
                
                gr.Markdown("""
                ### ℹ️ About
                This model uses a neural network trained on historical Porter delivery data 
                to estimate delivery times based on order characteristics, delivery partner 
                availability, and estimated driving distance.
                
                **Model Architecture:** 64→32→16→1 dense layers with ReLU activation, 
                dropout regularization, Adam optimizer, MSE loss.
                
                **Features:** Market ID, order details (items, prices), dasher availability, 
                and estimated driving duration.
                """)
        
        # Example inputs
        gr.Examples(
            examples=[
                [1, 4, 300, 3, 50, 150, 20, 5, 10, 15],
                [2, 2, 150, 2, 40, 100, 10, 8, 15, 25],
                [3, 6, 500, 4, 60, 200, 25, 3, 5, 10],
            ],
            inputs=[market_id, total_items, subtotal, num_distinct_items, min_item_price, 
                   max_item_price, total_onshift_dashers, total_busy_dashers, 
                   total_outstanding_orders, estimated_store_to_consumer_driving_duration],
            outputs=output,
            fn=predict_delivery_time,
            cache_examples=False,
        )
        
        predict_btn.click(
            fn=predict_delivery_time,
            inputs=[market_id, total_items, subtotal, num_distinct_items, min_item_price, 
                   max_item_price, total_onshift_dashers, total_busy_dashers, 
                   total_outstanding_orders, estimated_store_to_consumer_driving_duration],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
