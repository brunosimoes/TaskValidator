import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import tensorflow as tf
from matplotlib import pyplot as plt

class MetricsCalculator:
    """Utility class for calculating model evaluation metrics."""
    
    @staticmethod
    def calculate_classification_metrics(predictions, ground_truth, threshold=0.5):
        """
        Calculate binary classification metrics.
        
        Args:
            predictions (np.ndarray): Model predictions
            ground_truth (np.ndarray): Ground truth labels
            threshold (float): Classification threshold
            
        Returns:
            dict: Dictionary of metrics
        """
        # Convert to binary predictions
        binary_preds = predictions > threshold
        binary_truth = ground_truth > threshold
        
        # Calculate TP, FP, TN, FN
        tp = np.sum((binary_preds == True) & (binary_truth == True))
        fp = np.sum((binary_preds == True) & (binary_truth == False))
        tn = np.sum((binary_preds == False) & (binary_truth == False))
        fn = np.sum((binary_preds == False) & (binary_truth == True))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    @staticmethod
    def calculate_per_step_metrics(predictions, ground_truth, threshold=0.5):
        """
        Calculate metrics for each step in multi-step classification.
        
        Args:
            predictions (np.ndarray): Model predictions with shape (samples, num_steps)
            ground_truth (np.ndarray): Ground truth labels with shape (samples, num_steps)
            threshold (float): Classification threshold
            
        Returns:
            list: List of metric dictionaries for each step
        """
        # Apply threshold
        binary_preds = predictions > threshold
        binary_truth = ground_truth > threshold
        
        # Calculate metrics for each step
        per_step_metrics = []
        for step_idx in range(ground_truth.shape[1]):
            step_preds = binary_preds[:, step_idx]
            step_truth = binary_truth[:, step_idx]
            
            # Calculate metrics for this step
            metrics = MetricsCalculator.calculate_classification_metrics(
                predictions[:, step_idx], 
                ground_truth[:, step_idx], 
                threshold
            )
            
            per_step_metrics.append(metrics)
        
        return per_step_metrics
    
    @staticmethod
    def plot_confusion_matrices(predictions, ground_truth, threshold=0.5, step_names=None,
                              save_path=None, figsize=(15, 5)):
        """
        Plot confusion matrices for multi-step classification.
        
        Args:
            predictions (np.ndarray): Model predictions with shape (samples, num_steps)
            ground_truth (np.ndarray): Ground truth labels with shape (samples, num_steps)
            threshold (float): Classification threshold
            step_names (list, optional): Names for each step
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.Figure: The figure object
        """
        num_steps = ground_truth.shape[1]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Apply threshold
        binary_preds = predictions > threshold
        binary_truth = ground_truth > threshold
        
        # Plot confusion matrix for each step
        for step_idx in range(num_steps):
            plt.subplot(1, num_steps, step_idx + 1)
            
            step_preds = binary_preds[:, step_idx]
            step_truth = binary_truth[:, step_idx]
            
            # Calculate confusion matrix
            tp = np.sum((step_preds == True) & (step_truth == True))
            fp = np.sum((step_preds == True) & (step_truth == False))
            tn = np.sum((step_preds == False) & (step_truth == False))
            fn = np.sum((step_preds == False) & (step_truth == True))
            
            # Create confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Plot confusion matrix
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            
            # Add title
            if step_names and step_idx < len(step_names):
                plt.title(step_names[step_idx])
            else:
                plt.title(f'Step {step_idx + 1}')
                
            plt.colorbar()
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.xticks([0, 1], ['Neg', 'Pos'])
            plt.yticks([0, 1], ['Neg', 'Pos'])
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig

class DataProcessor:
    """Utility class for preprocessing data."""
    
    @staticmethod
    def preprocess_image(image_path, target_size=(224, 224)):
        """
        Preprocess a single image file.
        
        Args:
            image_path (str): Path to image file
            target_size (tuple): Size to resize the image to
            
        Returns:
            np.ndarray: Preprocessed image with shape (1, target_size[0], target_size[1], 3)
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
    
    @staticmethod
    def preprocess_sensor_data(data, sensor_columns, scaler=None, window_size=100):
        """
        Preprocess sensor data for model input.
        
        Args:
            data (pd.DataFrame or str): Sensor data or path to CSV file
            sensor_columns (list): List of sensor column names
            scaler (dict, optional): Dictionary of scaling parameters
            window_size (int): Window size for time series data
            
        Returns:
            np.ndarray: Preprocessed data with shape (1, window_size, len(sensor_columns))
        """
        # Load data if path is provided
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Extract relevant columns
        data = data[sensor_columns]
        
        # Apply scaling if provided
        if scaler:
            for col in sensor_columns:
                if col in scaler:
                    mean = scaler[col]['mean']
                    std = scaler[col]['std']
                    data[col] = (data[col] - mean) / std
        
        # Extract values
        sensor_data = data.values
        
        # Create window
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

class TrainingHistoryPlotter:
    """Utility class for plotting training history."""
    
    @staticmethod
    def plot_training_history(history, model_name, output_dir=None):
        """
        Plot training history for a model.
        
        Args:
            history (tf.keras.callbacks.History): Training history object
            model_name (str): Name of the model
            output_dir (str, optional): Directory to save plots
            
        Returns:
            matplotlib.Figure: The figure object
        """
        # Create figure
        fig = plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        plt.title(f'{model_name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'])
            plt.legend(['Train', 'Validation'], loc='lower right')
        else:
            plt.legend(['Train'], loc='lower right')
        plt.title(f'{model_name} Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        
        # Save the figure if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{model_name}_history.png'))
            
            # Save the history to a JSON file
            with open(os.path.join(output_dir, f'{model_name}_history.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {}
                for key, value in history.history.items():
                    history_dict[key] = [float(v) for v in value]
                json.dump(history_dict, f, indent=2)
        
        return fig

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image file for model input.
    
    Args:
        image_path (str): Path to image file
        target_size (tuple): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    return preprocess_image_array(img, target_size)

def preprocess_image_array(img, target_size=(224, 224)):
    """
    Preprocess an image array for model input.
    
    Args:
        img (np.ndarray): Input image array
        target_size (tuple): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img / 255.0
    
    return img

def preprocess_sensor_data(data, sensor_columns, scaler=None, window_size=100):
    """
    Preprocess sensor data for model input.
    
    Args:
        data (pd.DataFrame or str): Sensor data or path to CSV file
        sensor_columns (list): List of sensor column names
        scaler (object, optional): Scaler for normalization
        window_size (int): Window size for time series
        
    Returns:
        np.ndarray: Preprocessed sensor data
    """
    # Load data if path is provided
    if isinstance(data, str):
        if not os.path.exists(data):
            raise ValueError(f"Sensor data file not found: {data}")
        df = pd.read_csv(data)
    else:
        df = data
    
    # Check required columns
    missing_columns = [col for col in sensor_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Apply scaling if provided
    if scaler is not None:
        df[sensor_columns] = scaler.transform(df[sensor_columns])
    
    # Extract sensor data
    sensor_data = df[sensor_columns].values
    
    # Create windows
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

def load_vision_data(data_dir):
    """
    Load and preprocess vision data for training.
    
    Args:
        data_dir (str): Directory containing the vision dataset
        
    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val
    """
    print(f"Loading vision data from {data_dir}")
    
    # Look for vision data in different possible directories
    vision_dir = None
    possible_locations = [
        os.path.join(data_dir, 'vision'),
        data_dir,
        os.path.join(data_dir, 'vision', 'train'),
        os.path.join(data_dir, 'train')
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            vision_dir = location
            break
    
    if not vision_dir:
        raise ValueError(f"Could not find vision data in {data_dir}")
    
    # Look for training and validation datasets
    train_dir = os.path.join(vision_dir, 'train')
    val_dir = os.path.join(vision_dir, 'val')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    
    # Load training data
    X_train, y_train = _load_image_data(train_dir)
    
    # Load validation data if available
    X_val, y_val = None, None
    if os.path.exists(val_dir):
        X_val, y_val = _load_image_data(val_dir)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }

def _load_image_data(data_dir, input_shape=(224, 224)):
    """
    Load image data from a directory with step subdirectories.
    
    Args:
        data_dir (str): Directory containing step subdirectories
        input_shape (tuple): Target image dimensions
        
    Returns:
        tuple: (X, y) with image data and labels
    """
    print(f"Loading images from {data_dir}")
    
    # Find all step directories
    steps = []
    for item in os.listdir(data_dir):
        if item.startswith('step_') and os.path.isdir(os.path.join(data_dir, item)):
            steps.append(item)
    
    if not steps:
        raise ValueError(f"No step directories found in {data_dir}")
    
    # Sort steps by number
    steps.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(steps)} step directories: {steps}")
    
    # Collect images and labels
    X = []  # Images
    y = []  # Labels (one-hot encoded)
    
    for step_idx, step_dir in enumerate(steps):
        step_path = os.path.join(data_dir, step_dir)
        
        # Check for 'complete' and 'incomplete' subdirectories
        complete_dir = os.path.join(step_path, 'complete')
        incomplete_dir = os.path.join(step_path, 'incomplete')
        
        if not os.path.exists(complete_dir) or not os.path.exists(incomplete_dir):
            print(f"Warning: Missing complete/incomplete directories for {step_dir}")
            continue
        
        # Process 'complete' images
        print(f"Processing 'complete' images for {step_dir}...")
        for img_file in tqdm(os.listdir(complete_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(complete_dir, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_shape)
                img = img / 255.0  # Normalize
                
                X.append(img)
                
                # Create label with step_idx set to 1 (completed)
                label = np.zeros(len(steps))
                label[step_idx] = 1
                y.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Process 'incomplete' images
        print(f"Processing 'incomplete' images for {step_dir}...")
        for img_file in tqdm(os.listdir(incomplete_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(incomplete_dir, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_shape)
                img = img / 255.0  # Normalize
                
                X.append(img)
                
                # Create label with step_idx set to 0 (not completed)
                label = np.zeros(len(steps))
                # Previous steps are completed
                for i in range(step_idx):
                    label[i] = 1
                y.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images with shape {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def load_sensor_data(data_dir):
    """
    Load and preprocess sensor data for training.
    
    Args:
        data_dir (str): Directory containing the sensor dataset
        
    Returns:
        dict: Dictionary containing X_train, y_train, X_val, y_val
    """
    print(f"Loading sensor data from {data_dir}")
    
    # Look for sensor data files
    train_sensor_path = None
    val_sensor_path = None
    
    # Look for training data in various possible locations
    train_possible_locations = [
        os.path.join(data_dir, 'sensor', 'train_sensor_data.csv'),
        os.path.join(data_dir, 'train_sensor_data.csv'),
        os.path.join(data_dir, 'sensor_data.csv')
    ]
    
    for location in train_possible_locations:
        if os.path.exists(location):
            train_sensor_path = location
            break
    
    # Look for validation data
    val_possible_locations = [
        os.path.join(data_dir, 'sensor', 'val_sensor_data.csv'),
        os.path.join(data_dir, 'val_sensor_data.csv')
    ]
    
    for location in val_possible_locations:
        if os.path.exists(location):
            val_sensor_path = location
            break
    
    if not train_sensor_path:
        # Try to find any CSV file in the dataset directory as a last resort
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    train_sensor_path = os.path.join(root, file)
                    break
            if train_sensor_path:
                break
    
    if not train_sensor_path:
        raise ValueError(f"Could not find sensor data in {data_dir}")
    
    # Load training data
    train_df = pd.read_csv(train_sensor_path)
    print(f"Loaded training data from {train_sensor_path}, shape: {train_df.shape}")
    
    # Identify sensor columns and target columns
    sensor_columns = []
    for col in train_df.columns:
        if col in ['pressure', 'temperature', 'proximity', 'vibration']:
            sensor_columns.append(col)
    
    if not sensor_columns:
        raise ValueError(f"No sensor columns found in {train_sensor_path}")
    
    print(f"Identified sensor columns: {sensor_columns}")
    
    # Prepare data for LSTM (time windows)
    window_size = 100
    step_size = 20
    
    X_train, y_train = _prepare_sensor_sequences(
        train_df, 
        sensor_columns, 
        window_size, 
        step_size
    )
    
    # Load validation data if available
    X_val, y_val = None, None
    if val_sensor_path:
        val_df = pd.read_csv(val_sensor_path)
        print(f"Loaded validation data from {val_sensor_path}, shape: {val_df.shape}")
        
        X_val, y_val = _prepare_sensor_sequences(
            val_df, 
            sensor_columns, 
            window_size, 
            step_size
        )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'sensor_columns': sensor_columns
    }

def _prepare_sensor_sequences(df, sensor_columns, window_size=100, step_size=20):
    """
    Prepare sensor data sequences for LSTM training.
    
    Args:
        df (pd.DataFrame): Sensor data DataFrame
        sensor_columns (list): List of sensor column names
        window_size (int): Window size for time series
        step_size (int): Step size for sliding window
        
    Returns:
        tuple: (X, y) with sensor data sequences and labels
    """
    # Ensure sensor_columns exist in the dataframe
    available_columns = [col for col in sensor_columns if col in df.columns]
    
    if not available_columns:
        raise ValueError(f"None of the specified sensor columns {sensor_columns} found in dataframe. Available columns: {df.columns.tolist()}")
    
    if len(available_columns) != len(sensor_columns):
        print(f"Warning: Only {len(available_columns)} of {len(sensor_columns)} sensor columns found in dataframe.")
        print(f"Using available columns: {available_columns}")
    
    # Normalize sensor data
    for col in available_columns:
        mean = df[col].mean()
        std = df[col].std() or 1.0  # Avoid division by zero
        df[col] = (df[col] - mean) / std
       
    # Check for step_id and completed columns
    has_step_id = 'step_id' in df.columns
    has_completed = 'completed' in df.columns
    
    # Create sequences
    X = []
    y = []
    
    if has_step_id:
        # Group by step_id to ensure sequences don't cross step boundaries
        for step_id, group in df.groupby('step_id'):
            sensor_data = group[sensor_columns].values
            completed = group['completed'].values if has_completed else np.ones(len(sensor_data))
            
            # Create sliding windows
            for i in range(0, len(sensor_data) - window_size + 1, step_size):
                X.append(sensor_data[i:i+window_size])
                
                # Create target label based on current step and completion status
                label = np.zeros(df['step_id'].max() + 1)
                
                # Mark current step as completed if the last frame is completed
                if completed[i+window_size-1]:
                    label[step_id] = 1
                
                # Mark previous steps as completed (assuming sequential steps)
                for j in range(step_id):
                    label[j] = 1
                    
                y.append(label)
    else:
        # No step information, treat as single sequence
        sensor_data = df[sensor_columns].values
        
        # Create sliding windows
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            X.append(sensor_data[i:i+window_size])
            
            # Since we don't have step info, create a dummy label
            # You'll need to modify this based on your actual requirements
            y.append(np.array([1]))  # Dummy label
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    
    return X, y