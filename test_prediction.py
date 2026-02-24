"""
Test script to demonstrate the prediction functionality.
"""

import pandas as pd
import numpy as np
import os

# Test prediction functionality
def test_prediction():
    print("Testing prediction functionality...")
    
    # Check if we have a trained model
    model_path = "models/best_RandomForest_best_model.pkl"
    
    if not os.path.exists(model_path):
        print("No trained model found. Please run the main pipeline first.")
        return
    
    # Create a simple test dataset (using a small sample from the original dataset)
    # For demonstration, we'll use a small sample from the test data
    try:
        # Load a small sample from the original dataset
        df = pd.read_csv("data/Friday-WorkingHours-Morning.pcap_ISCX.csv", nrows=10)
        
        # Save as test file
        test_file = "test_sample.csv"
        df.to_csv(test_file, index=False)
        print(f"Created test file: {test_file}")
        
        # Run prediction
        from src.predict import predict_new_data
        results = predict_new_data(model_path, test_file)
        
        if results['success']:
            print("\nPrediction Results:")
            print(f"Input samples: {results['input_shape'][0]}")
            print(f"Predictions: {results['predictions']}")
            if results['probabilities'] is not None:
                print(f"Confidence scores: {np.max(results['probabilities'], axis=1)}")
        else:
            print("Prediction failed")
            
    except Exception as e:
        print(f"Error during prediction test: {str(e)}")

if __name__ == "__main__":
    test_prediction()