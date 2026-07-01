import os
import requests

DATA_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
DATA_PATH = "data/Mall_Customers.csv"

def download_data():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        if response.status_code == 200:
            with open(DATA_PATH, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print("Failed to download dataset. Please check the URL or your internet connection.")
            print(f"Status code: {response.status_code}")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_data()
