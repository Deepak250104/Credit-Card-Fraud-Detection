import kagglehub
import os

# Define the path where the dataset should be downloaded
download_path = 'Data/'

# Ensure the download directory exists
os.makedirs(download_path, exist_ok=True)

try:
    # Download the latest version of the dataset to the specified path
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud", path=download_path, force=False)
    # force=False will only download if the dataset doesn't exist or a new version is available

    print("Dataset downloaded successfully to:", path)

except Exception as e:
    print(f"An error occurred during dataset download: {e}")
    print("Please ensure you have the kagglehub library installed (`pip install kagglehub`)")
    print("and that you are authenticated with Kaggle (you might need a Kaggle API key).")