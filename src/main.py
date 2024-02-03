# main.py
import os
import sys

# Update your imports to use the package structure
from src.data_preprocessing.data_preprocessing import main as preprocess_main
from src.train.train import main as train_main
from src.inference.run_inference import main as inference_main
from src.eda import main as eda_main
from src.utils import get_data_path, get_output_path, get_logger

def main():
    # Step 1: Exploratory Data Analysis
    print("Starting EDA...")
    eda_main()

    # Step 2: Data Preprocessing
    print("Starting data preprocessing...")
    preprocess_main()

    # Step 3: Training
    print("Starting training process...")
    train_main()

    # Step 4: Inference
    print("Running inference...")
    inference_main()

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
