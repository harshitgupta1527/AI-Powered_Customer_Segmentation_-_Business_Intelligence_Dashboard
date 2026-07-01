import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_data(file_path="data/Mall_Customers.csv"):
    """Loads the dataset from the specified path."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

def get_data_summary(df):
    """Returns a dictionary containing summary statistics of the dataset."""
    summary = {
        "shape": df.shape,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe().to_dict()
    }
    return summary

def clean_and_preprocess(df, models_dir="models"):
    """
    Cleans the data, encodes categorical variables, and scales numerical features.
    Saves the scaler and encoder to models_dir.
    """
    df_clean = df.copy()
    
    # Handle duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values (if any, though Mall Customers typically has none)
    df_clean = df_clean.dropna()

    # Encode Gender
    label_encoder = LabelEncoder()
    if 'Gender' in df_clean.columns:
        df_clean['Gender_Encoded'] = label_encoder.fit_transform(df_clean['Gender'])
    
    # Features for clustering (typically we use Annual Income and Spending Score, or Age as well)
    # Let's use Age, Annual Income, and Spending Score for 3D clustering
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Verify features exist
    for f in features:
        if f not in df_clean.columns:
            raise ValueError(f"Feature {f} not found in dataset.")

    X = df_clean[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save models
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))

    return df_clean, X_scaled, features
