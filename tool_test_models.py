import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the vision model
    
    Parameters:
    image_path (str): Path to the image file
    target_size (tuple): Target size for resizing
    
    Returns:
    numpy.ndarray: Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def preprocess_sensor_data(sensor_data, scaler_path=None, window_size=100):
    """
    Preprocess sensor data for the sensor model
    
    Parameters:
    sensor_data (str or pandas.DataFrame): Path to CSV file or DataFrame
    scaler_path (str): Path to scaler JSON file
    window_size (int): Window size for the model
    
    Returns:
    numpy.ndarray: Preprocessed sensor data
    """
    # Load data
    if isinstance(sensor_data, str):
        if not os.path.exists(sensor_data):
            raise ValueError(f"Sensor data file not found: {sensor_data}")
        df = pd.read_csv(sensor_data)
    else:
        df = sensor_data
    
    # Get sensor columns
    sensor_columns = [col for col in df.columns if col in ['pressure', 'temperature', 'proximity', 'vibration']]
    if not sensor_columns:
        raise ValueError("No sensor columns found in the data")
    
    # Apply scaling if provided
    if scaler_path is not None and os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            scalers = json.load(f)
        
        for col in sensor_columns:
            if col in scalers:
                mean = scalers[col]['mean']
                std = scalers[col]['std']
                df[col] = (df[col] - mean) / std
    
    # Create a window of data
    sensor_data = df[sensor_columns].values
    if len(sensor_data) < window_size:
        # Pad data if not enough samples
        pad_rows = window_size - len(sensor_data)
        padding = np.zeros((pad_rows, len(sensor_columns)))
        sensor_data = np.vstack([padding, sensor_data])
    else:
        # Take the last window_size samples
        sensor_data = sensor_data[-window_size:]
    
    # Add batch dimension
    sensor_data = np.expand_dims(sensor_data, axis=0)
    
    return sensor_data

def test_vision_model(model_path, image_path, config, output_dir=None):
    """
    Test the vision model on a single image, using VisionProcessor if available
    
    Parameters:
    model_path (str): Path to the trained model
    image_path (str): Path to the image file
    config (dict): Configuration with step definitions
    output_dir (str): Directory to save results
    
    Returns:
    dict: Prediction results
    """
    print(f"Testing vision model on image: {image_path}")
    
    # Get step definitions
    step_definitions = config.get("step_definitions", [])
    
    try:
        # Try to use VisionProcessor first
        from app_vision_processor import VisionProcessor
        
        # Determine number of steps from step_definitions
        num_steps = len(step_definitions)
        if num_steps == 0:
            # If no step definitions, try to infer from model structure
            # For now, default to a reasonable number
            num_steps = 5
            print("Warning: No step definitions found, assuming 5 steps")
        
        # Create VisionProcessor instance
        vision_processor = VisionProcessor(num_process_steps=num_steps, model_path=model_path)
        
        # Load and preprocess image using VisionProcessor's method
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Use VisionProcessor's detect_steps method
        result_dict = vision_processor.detect_steps(img)
        
        # Format results to match expected output structure
        predictions = result_dict['predictions']
        
    except (ImportError, Exception) as e:
        print(f"Could not use VisionProcessor: {e}")
        print("Falling back to standard TensorFlow loading")
        
        # Load model using standard method
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded vision model from {model_path}")
        
        # Use original preprocess_image function
        img = preprocess_image(image_path)
        
        # Run prediction
        predictions = model.predict(img)[0]

    # Format results
    results = {
        "predictions": {},
        "completed_steps": []
    }
    
    for i, pred in enumerate(predictions):
        step_name = step_definitions[i] if i < len(step_definitions) else f"Step {i}"
        results["predictions"][step_name] = float(pred)
        if pred > 0.5:
            results["completed_steps"].append(step_name)
    
    # Print results
    print("\nVision Model Predictions:")
    print(f"Image: {os.path.basename(image_path)}")
    print("\nStep Completion Probabilities:")
    for step, prob in results["predictions"].items():
        status = "COMPLETE" if prob > 0.5 else "INCOMPLETE"
        print(f"  - {step}: {prob:.4f} ({status})")
    
    print("\nCompleted Steps:")
    if results["completed_steps"]:
        for step in results["completed_steps"]:
            print(f"  - {step}")
    else:
        print("  None")
    
    # Visualize results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image for visualization
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        
        # Show predictions as bar chart
        plt.subplot(1, 2, 2)
        steps = list(results["predictions"].keys())
        probs = list(results["predictions"].values())
        colors = ['green' if p > 0.5 else 'red' for p in probs]
        
        plt.barh(steps, probs, color=colors)
        plt.xlabel('Probability')
        plt.title('Step Completion Predictions')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add threshold line
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vision_prediction.png'))
        print(f"\nVisualization saved to {os.path.join(output_dir, 'vision_prediction.png')}")
        
        # Save results to JSON file
        results_json_path = os.path.join(output_dir, 'results.json')
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_json_path}")
    
    return results

def test_sensor_model(model_path, sensor_data, config, scaler_path=None, output_dir=None):
    """
    Test the sensor model on sensor data
    
    Parameters:
    model_path (str): Path to the trained model
    sensor_data (str): Path to sensor data CSV file
    config (dict): Configuration with step definitions
    scaler_path (str): Path to scaler JSON file
    output_dir (str): Directory to save results
    
    Returns:
    dict: Prediction results
    """
    print(f"Testing sensor model on data: {sensor_data}")
    
    # Load model
    model = load_model(model_path)
    print(f"Loaded sensor model from {model_path}")
    
    # Get step definitions
    step_definitions = config.get("step_definitions", [])
    if not step_definitions:
        print("Warning: No step definitions found in config")
        step_definitions = [f"Step {i}" for i in range(model.output_shape[-1])]
    
    # Preprocess sensor data
    processed_data = preprocess_sensor_data(sensor_data, scaler_path)
    
    # Run prediction
    predictions = model.predict(processed_data)[0]
    
    # Format results
    results = {
        "predictions": {},
        "completed_steps": []
    }
    
    for i, pred in enumerate(predictions):
        step_name = step_definitions[i] if i < len(step_definitions) else f"Step {i}"
        results["predictions"][step_name] = float(pred)
        if pred > 0.5:
            results["completed_steps"].append(step_name)
    
    # Print results
    print("\nSensor Model Predictions:")
    print("\nStep Completion Probabilities:")
    for step, prob in results["predictions"].items():
        status = "COMPLETE" if prob > 0.5 else "INCOMPLETE"
        print(f"  - {step}: {prob:.4f} ({status})")
    
    print("\nCompleted Steps:")
    if results["completed_steps"]:
        for step in results["completed_steps"]:
            print(f"  - {step}")
    else:
        print("  None")
    
    # Visualize results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data for visualization
        if isinstance(sensor_data, str):
            df = pd.read_csv(sensor_data)
        else:
            df = sensor_data
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot sensor data
        plt.subplot(2, 1, 1)
        sensor_columns = [col for col in df.columns if col in ['pressure', 'temperature', 'proximity', 'vibration']]
        for col in sensor_columns:
            plt.plot(df[col].values, label=col)
        
        plt.title('Sensor Data')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show predictions as bar chart
        plt.subplot(2, 1, 2)
        steps = list(results["predictions"].keys())
        probs = list(results["predictions"].values())
        colors = ['green' if p > 0.5 else 'red' for p in probs]
        
        plt.barh(steps, probs, color=colors)
        plt.xlabel('Probability')
        plt.title('Step Completion Predictions')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add threshold line
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sensor_prediction.png'))
        print(f"\nVisualization saved to {os.path.join(output_dir, 'sensor_prediction.png')}")
        
        # Save results to JSON file
        results_json_path = os.path.join(output_dir, 'results.json')
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_json_path}")
    
    return results

def integrate_predictions(vision_results, sensor_results):
    """
    Integrate vision and sensor model predictions
    
    Parameters:
    vision_results (dict): Vision model prediction results
    sensor_results (dict): Sensor model prediction results
    
    Returns:
    dict: Integrated prediction results
    """
    if vision_results is None and sensor_results is None:
        return None
    
    # If only one model was used, return its results
    if vision_results is None:
        return sensor_results
    if sensor_results is None:
        return vision_results
    
    # Get all steps from both models
    all_steps = set(vision_results["predictions"].keys()).union(
        set(sensor_results["predictions"].keys())
    )
    
    # Combine predictions
    combined_results = {
        "predictions": {},
        "completed_steps": []
    }
    
    for step in all_steps:
        vision_prob = vision_results["predictions"].get(step, 0)
        sensor_prob = sensor_results["predictions"].get(step, 0)
        
        # Weighted average (vision 60%, sensor 40%)
        combined_prob = 0.6 * vision_prob + 0.4 * sensor_prob
        combined_results["predictions"][step] = combined_prob
        
        if combined_prob > 0.5:
            combined_results["completed_steps"].append(step)
    
    return combined_results

def main():
    parser = argparse.ArgumentParser(description='Test trained task ML models')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--vision-model', help='Path to trained vision model')
    parser.add_argument('--sensor-model', help='Path to trained sensor model')
    parser.add_argument('--image', help='Path to test image')
    parser.add_argument('--sensor-data', help='Path to test sensor data CSV file')
    parser.add_argument('--scaler', help='Path to sensor scaler JSON file')
    parser.add_argument('--output', default='test_results', help='Path to save test results')
    args = parser.parse_args()
    
    # Check if at least one model is provided
    if not args.vision_model and not args.sensor_model:
        parser.error("At least one of --vision-model or --sensor-model must be provided")
    
    # Check if corresponding test data is provided
    if args.vision_model and not args.image:
        parser.error("--image must be provided when testing vision model")
    if args.sensor_model and not args.sensor_data:
        parser.error("--sensor-data must be provided when testing sensor model")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Test models
    vision_results = None
    sensor_results = None
    
    if args.vision_model and args.image:
        vision_results = test_vision_model(args.vision_model, args.image, config, args.output)
    
    if args.sensor_model and args.sensor_data:
        sensor_results = test_sensor_model(args.sensor_model, args.sensor_data, config, args.scaler, args.output)
    
    # Integrate results if both models were tested
    if vision_results is not None and sensor_results is not None:
        combined_results = integrate_predictions(vision_results, sensor_results)
        
        print("\n=== Combined Model Predictions ===")
        print("\nIntegrated Step Completion Probabilities:")
        for step, prob in combined_results["predictions"].items():
            status = "COMPLETE" if prob > 0.5 else "INCOMPLETE"
            print(f"  - {step}: {prob:.4f} ({status})")
        
        print("\nCompleted Steps (Combined):")
        if combined_results["completed_steps"]:
            for step in combined_results["completed_steps"]:
                print(f"  - {step}")
        else:
            print("  None")
        
        # Save combined results
        combined_results_path = os.path.join(args.output, 'results.json')
        with open(combined_results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\nCombined results saved to {combined_results_path}")
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()