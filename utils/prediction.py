import joblib
import os
import pandas as pd
import numpy as np

def load_models(models_dir="models"):
    """Loads the scaler and kmeans model from disk."""
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    kmeans_path = os.path.join(models_dir, 'kmeans_model.pkl')
    
    if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
        return None, None
        
    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    return scaler, kmeans

def predict_customer_segment(age, income, spending_score, scaler, kmeans, personas_map=None):
    """
    Predicts the cluster for a new customer.
    Features order must match training: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    """
    # Create input dataframe matching training features
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending_score]
    })
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    cluster = kmeans.predict(input_scaled)[0]
    
    # Get persona if map is provided
    persona = personas_map.get(cluster, "Standard Customer") if personas_map else f"Cluster {cluster}"
    
    # Confidence (distance to cluster center compared to others)
    distances = kmeans.transform(input_scaled)[0]
    min_dist = distances[cluster]
    avg_dist = np.mean(distances)
    # Simple confidence proxy: the smaller the distance compared to average, the higher the confidence
    confidence = max(0.0, min(100.0, (1 - (min_dist / avg_dist)) * 100 + 20))
    
    # Business Recommendation
    recommendation = ""
    if "High Value" in persona:
        recommendation = "Target with exclusive VIP offers and premium tier products."
    elif "Careful" in persona:
        recommendation = "Send targeted discount codes and value-based marketing."
    elif "Careless" in persona:
        recommendation = "Promote trendy, impulse-buy items and affordable luxury."
    elif "Sensible" in persona:
        recommendation = "Focus on essentials, bundle deals, and budget-friendly items."
    else:
        recommendation = "Engage with standard newsletter and seasonal promotions."
        
    return {
        'cluster': cluster,
        'persona': persona,
        'confidence': confidence,
        'recommendation': recommendation
    }
