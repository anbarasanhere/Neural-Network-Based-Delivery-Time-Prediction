# Neural-Network-Based-Delivery-Time-Prediction
# 📦 Food Delivery Time Prediction

## 📌 Project Overview

This project focuses on predicting the **estimated delivery time for intra-city food orders** using machine learning techniques. The goal is to improve operational efficiency and customer experience by leveraging structured data such as order timestamps, restaurant attributes, delivery partner availability, and order composition.

The problem is modeled as a **regression task**, where the target variable is the total delivery duration from order placement to completion.

---

## 🎯 Problem Statement

Accurately estimate delivery times for food orders within a city using features such as:

* Order creation time
* Restaurant category
* Number of delivery partners available
* Order details (e.g., item count, subtotal)

This prediction helps reduce uncertainty in delivery logistics and enhances real-time decision-making.

---

## 🚀 Applications

This model and its extensions can be applied across multiple domains:

* **Logistics Optimization**
  Improve delivery routing and partner allocation.

* **Dynamic Pricing**
  Adjust delivery fees based on predicted demand and delays.

* **Customer Experience**
  Provide accurate real-time ETAs in food delivery platforms.

* **Inventory & Kitchen Management**
  Help restaurants optimize preparation timing based on delivery conditions.

---

## 🕒 Working with Time Data

Handling temporal data is critical for this project.

### Key Concepts

* **Datetime**
  Represents a specific timestamp (e.g., `2023-10-05 14:30:00`)

* **Timedelta**
  Represents duration between two timestamps (e.g., delivery time)

* **Period (Time Span)**
  Represents a span of time (e.g., a day, month, or quarter)

### Useful Pandas Functions

* `dt.hour` → Extracts the hour from a timestamp
* `dt.dayofweek` → Returns the day index (0 = Monday)
* `dt.to_period('D')` → Converts timestamp to daily period

---

## ⚠️ Outlier Handling

Outliers significantly impact model performance and evaluation metrics.

### Why Remove Outliers?

* Distort training and introduce noise
* Skew metrics like MAE, MSE, RMSE
* Reduce generalization capability

**Observed Impact:**

* MAE reduced from **47 minutes → 12.6 minutes** after outlier removal

### Methods Used

* **IQR (Interquartile Range)**
* **Z-Score Filtering**
* **Winsorization**

---

## 🤖 Modeling Approaches

### Classical Machine Learning

* **Linear Regression**
  Baseline model for interpretability

* **Decision Trees / Random Forest**
  Capture non-linear relationships

* **Gradient Boosting (XGBoost / LightGBM)**
  Strong performance on structured/tabular data

---

## 🧠 Neural Network Approach

### Why Neural Networks?

* Capture complex feature interactions
* Perform better with large-scale datasets

### Key Considerations

#### Feature Scaling

Scaling ensures stable and efficient training.

* Prevents uneven gradient updates
* Improved MSE significantly after scaling

#### Optimizer

* **Adam Optimizer**

  * Combines momentum and adaptive learning rates
  * Handles noisy data effectively
  * Reduces need for manual tuning

#### Activation Functions

* **Hidden Layers:** ReLU

  * Avoids vanishing gradient problem
  * Efficient for deep networks

* **Output Layer:** Linear

  * Suitable for regression tasks

---

## 📊 Model Performance Insights

* Neural network achieved limited performance (**R² ≈ 0.2**)
* Indicates:

  * Possible data limitations
  * Need for feature engineering or larger dataset

---

## 📈 Future Improvements

* Incorporate real-time traffic and weather data
* Experiment with ensemble models
* Increase dataset size for better generalization
* Feature engineering (interaction terms, lag features)

---

## 🧩 Conclusion

This project demonstrates how data preprocessing, especially **time handling and outlier removal**, plays a critical role in predictive modeling. While neural networks offer flexibility, **tree-based models remain highly competitive for tabular data**.

The system can be extended into a production-grade solution for real-time delivery estimation and operational optimization.

---
