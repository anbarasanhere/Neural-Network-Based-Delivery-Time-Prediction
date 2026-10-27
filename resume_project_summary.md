# Porter Delivery Time Estimation - Resume Project Summary

## Project Title
**Porter Delivery Time Estimation using Neural Networks**

## Key Accomplishments (3-4 quantifiable points)

### 1. 🤖 Built & Trained Neural Network Model
- Developed a 3-layer dense neural network (64→32→16→1 architecture) using TensorFlow/Keras
- Trained on **140,621 samples** with **78.3% R² score** on test data
- Achieved **3.21 minutes Mean Absolute Error** (MAE) for delivery time predictions
- Optimized using Adam optimizer with MSE loss for regression task

### 2. 📊 Advanced Data Preprocessing & Feature Engineering
- Engineered **10 key features** from raw order data including: market ID, item counts, price ranges, dasher availability, and estimated driving duration
- Implemented **KNN Imputer** (k=5) for missing value handling
- Applied **StandardScaler** for feature normalization critical to neural network convergence
- Removed outliers using IQR method (5th-95th percentile range) improving model accuracy

### 3. 🔧 Full ML Pipeline Development
- End-to-end pipeline: data ingestion → preprocessing → model training → evaluation → deployment
- Used **TensorFlow Sequential model** with ReLU activation and dropout regularization
- Achieved **4.36 minutes RMSE** on test set, demonstrating robust prediction accuracy
- Log-transformed target variable (log1p) and exponentiated predictions (expm1) for interpretable results

### 4. 🚀 Deployable Web Interface
- Created **Gradio-based UI** enabling real-time ETA prediction from order features
- Model artifacts saved and loadable for production deployment
- Designed interactive interface with 10 input features for customer experimentation
- Project structure ready for Hugging Face Spaces deployment

## Technical Skills Demonstrated
- **Deep Learning**: TensorFlow 2.x, Keras Sequential models, dense layers, ReLU activation
- **ML Preprocessing**: StandardScaler, KNN Imputer, outlier removal, log transformation
- **Data Engineering**: Pandas data manipulation, datetime feature extraction, one-hot encoding
- **Model Evaluation**: MAE, RMSE, R² score, train-test split (80/20), validation split
- **Python ecosystem**: pandas, numpy, scikit-learn, tensorflow, gradio