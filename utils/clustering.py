from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import joblib
import os

def find_optimal_k(X_scaled, max_k=10):
    """
    Evaluates KMeans from k=2 to max_k to find optimal K.
    Returns inertias, silhouette scores, and davies-bouldin scores.
    """
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': []
    }
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        metrics['k'].append(k)
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X_scaled, kmeans.labels_))
        metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, kmeans.labels_))
        
    return pd.DataFrame(metrics)

def train_kmeans(X_scaled, n_clusters, models_dir="models"):
    """
    Trains the final KMeans model with the specified n_clusters.
    Saves the model to models_dir.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save model
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))
    
    return kmeans

def assign_personas(df, cluster_col='Cluster'):
    """
    Assigns business-friendly personas to clusters based on average income and spending.
    Note: The specific logic depends on the typical Mall Customers distribution.
    """
    # Calculate means per cluster for Income and Spending
    cluster_means = df.groupby(cluster_col)[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
    
    personas = {}
    for cluster, row in cluster_means.iterrows():
        income = row['Annual Income (k$)']
        spending = row['Spending Score (1-100)']
        
        if income > 70 and spending > 70:
            personas[cluster] = "High Value Customers"
        elif income > 70 and spending < 40:
            personas[cluster] = "Careful Spenders"
        elif income < 40 and spending > 70:
            personas[cluster] = "Careless Spenders"
        elif income < 40 and spending < 40:
            personas[cluster] = "Sensible Customers"
        else:
            personas[cluster] = "Standard Customers"
            
    df['Persona'] = df[cluster_col].map(personas)
    return df, personas

def generate_insights(personas):
    """Generates text insights based on assigned personas."""
    insights = []
    for cluster, persona in personas.items():
        if persona == "High Value Customers":
            insights.append(f"Cluster {cluster} represents **High Value Customers** (High Income, High Spending). Strategy: Target them with premium products and VIP programs.")
        elif persona == "Careful Spenders":
            insights.append(f"Cluster {cluster} represents **Careful Spenders** (High Income, Low Spending). Strategy: Offer targeted discounts or loyalty programs to encourage spending.")
        elif persona == "Careless Spenders":
            insights.append(f"Cluster {cluster} represents **Careless Spenders** (Low Income, High Spending). Strategy: Promote affordable luxury or trendy items.")
        elif persona == "Sensible Customers":
            insights.append(f"Cluster {cluster} represents **Sensible Customers** (Low Income, Low Spending). Strategy: Focus on budget-friendly and essential products.")
        else:
            insights.append(f"Cluster {cluster} represents **Standard Customers** (Average Income, Average Spending). Strategy: General marketing campaigns and volume discounts.")
    return insights
