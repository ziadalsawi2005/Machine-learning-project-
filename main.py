"""
Main execution script for Network Intrusion Detection System.

This script orchestrates the entire ML pipeline from data preprocessing
to model training, evaluation, and saving.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path to import modules
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from preprocessing_fixed import load_and_preprocess_data
from train import train_models
from evaluate import evaluate_model_performance, compare_models
from predict import predict_new_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input CSV dataset')
    parser.add_argument('--binary_classification', action='store_true',
                        help='Convert to binary classification (Normal vs Attack)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of test set (default: 0.2)')
    parser.add_argument('--metric', type=str, default='recall',
                        choices=['recall', 'f1_score', 'accuracy', 'precision'],
                        help='Metric to use for model selection (default: recall)')
    parser.add_argument('--model_path', type=str,
                        help='Path to saved model for prediction (optional)')
    parser.add_argument('--predict_file', type=str,
                        help='Path to CSV file for prediction (requires --model_path)')
    
    return parser.parse_args()


def run_complete_pipeline(args) -> Dict[str, Any]:
    """
    Run the complete ML pipeline.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    print("="*60)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("="*60)
    
    results = {}
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1] DATA PREPROCESSING")
    print("-"*30)
    
    try:
        data_dict = load_and_preprocess_data(
            data_path=args.data_path,
            binary_classification=args.binary_classification,
            test_size=args.test_size
        )
        print("[SUCCESS] Data preprocessing completed")
        results['preprocessing'] = 'completed'
    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {str(e)}")
        results['preprocessing'] = f'failed: {str(e)}'
        return results
    
    # Step 2: Model Training
    print("\n[STEP 2] MODEL TRAINING")
    print("-"*30)
    
    try:
        best_model_name, best_model, saved_paths = train_models(
            data_dict=data_dict,
            tune_hyperparams=args.tune_hyperparams,
            metric_for_selection=args.metric
        )
        print("[SUCCESS] Model training completed")
        results['training'] = {
            'best_model': best_model_name,
            'saved_paths': saved_paths
        }
    except Exception as e:
        print(f"[ERROR] Model training failed: {str(e)}")
        results['training'] = f'failed: {str(e)}'
        return results
    
    # Step 3: Model Evaluation
    print("\n[STEP 3] MODEL EVALUATION")
    print("-"*30)
    
    try:
        # Evaluate the best model
        eval_report = evaluate_model_performance(
            model=best_model,
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            model_name=best_model_name,
            class_names=getattr(data_dict.get('label_encoder'), 'classes_', None)
        )
        
        # Compare all models (need to get models from trainer)
        # Get the trainer from preprocessing data
        preprocessor = data_dict.get('preprocessor')
        if hasattr(preprocessor, 'trained_models'):
            models_dict = preprocessor.trained_models
        else:
            # If not available in data_dict, we need to retrain just for comparison
            from train import ModelTrainer
            trainer = ModelTrainer()
            trainer.initialize_models()
            models_dict = trainer.train_with_tuning(data_dict['X_train'], data_dict['y_train'], args.tune_hyperparams)
        
        comparison_results = compare_models(
            models_dict=models_dict,
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            class_names=getattr(data_dict.get('label_encoder'), 'classes_', None)
        )
        
        print("[SUCCESS] Model evaluation completed")
        results['evaluation'] = {
            'best_model_report': eval_report,
            'comparison_results': comparison_results
        }
    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {str(e)}")
        results['evaluation'] = f'failed: {str(e)}'
        return results
    
    # Summary
    print("\n[PIPELINE SUMMARY]")
    print("-"*30)
    print(f"Best model: {results['training']['best_model']}")
    print(f"Selected based on: {args.metric}")
    print(f"Binary classification: {args.binary_classification}")
    print(f"Hyperparameter tuning: {args.tune_hyperparams}")
    print(f"Test size: {args.test_size}")
    
    # Security-focused metrics from evaluation
    if 'evaluation' in results:
        best_model_metrics = results['evaluation']['best_model_report']['performance_metrics']
        print(f"\nBest Model Performance:")
        print(f"  Accuracy:  {best_model_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {best_model_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {best_model_metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score:  {best_model_metrics.get('f1_score', 'N/A'):.4f}")
        
        # Security implications
        security_analysis = results['evaluation']['best_model_report'].get('security_analysis', {})
        if 'false_negative_rate' in security_analysis:
            print(f"\nSecurity Analysis:")
            print(f"  False Negative Rate: {security_analysis['false_negative_rate']:.4f}")
            print(f"  False Positive Rate: {security_analysis['false_positive_rate']:.4f}")
    
    return results


def run_prediction_only(args) -> Dict[str, Any]:
    """
    Run prediction on new data using a saved model.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    print("="*60)
    print("NETWORK INTRUSION PREDICTION")
    print("="*60)
    
    if not args.model_path or not args.predict_file:
        print("[ERROR] Both --model_path and --predict_file are required for prediction")
        return {'success': False, 'error': 'Missing required arguments for prediction'}
    
    print(f"Loading model from: {args.model_path}")
    print(f"Predicting on file: {args.predict_file}")
    
    try:
        results = predict_new_data(
            model_path=args.model_path,
            input_file=args.predict_file,
            output_file=None  # We can add this as an option later
        )
        
        if results['success']:
            print("[SUCCESS] Prediction completed")
            print(f"Processed {results['input_shape'][0]} samples")
            
            # Show prediction summary
            unique, counts = np.unique(results['predictions'], return_counts=True)
            print("\nPrediction Distribution:")
            for label, count in zip(unique, counts):
                print(f"  {label}: {count} samples ({count/len(results['predictions'])*100:.2f}%)")
        else:
            print("[ERROR] Prediction failed")
        
        return results
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return {'success': False, 'error': str(e)}


def main():
    """Main function to orchestrate the pipeline."""
    args = parse_arguments()
    
    # Check if we're running prediction only
    if args.model_path and args.predict_file:
        return run_prediction_only(args)
    elif args.data_path:
        return run_complete_pipeline(args)
    else:
        print("Either --data_path (for training) or both --model_path and --predict_file (for prediction) must be provided.")
        return {'success': False, 'error': 'Invalid arguments'}


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED")
    print("="*60)