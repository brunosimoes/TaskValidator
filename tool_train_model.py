import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys

# Add current directory to path to find processor modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app_vision_processor import VisionProcessor
    vision_processor_available = True
    print("VisionProcessor module loaded successfully")
except ImportError:
    vision_processor_available = False
    print("VisionProcessor module not available, will use standard model architecture")

# Try to import SensorProcessor
try:
    from sensor_processor import SensorProcessor
    sensor_processor_available = True
    print("SensorProcessor module loaded successfully")
except ImportError:
    sensor_processor_available = False
    print("SensorProcessor module not available, will use standard model architecture")

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_image_data(data_dir, output_dir, input_shape=(224, 224), val_split=0.2):
    """
    Preprocess and organize image data for training
    
    Parameters:
    data_dir (str): Directory containing the generated image data
    output_dir (str): Directory to save processed data
    input_shape (tuple): Target image dimensions
    val_split (float): Validation split ratio
    
    Returns:
    tuple: (X_train, y_train, X_val, y_val) numpy arrays
    """
    print("Preprocessing image data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all step directories
    steps = []
    for item in os.listdir(data_dir):
        if item.startswith('step_') and os.path.isdir(os.path.join(data_dir, item)):
            steps.append(item)
    
    if not steps:
        print(f"No step directories found in {data_dir}")
        print(f"Available items: {os.listdir(data_dir)}")
        return None, None, None, None
    
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
            print(f"Available items: {os.listdir(step_path)}")
            continue
        
        # Process 'complete' images
        print(f"Processing 'complete' images for {step_dir}...")
        for img_file in tqdm(os.listdir(complete_dir)):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(complete_dir, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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
    
    print(f"Processed {len(X)} images with shape {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Save a few examples for verification
    for i in range(min(5, len(X_train))):
        img = X_train[i] * 255
        img = img.astype(np.uint8)
        label = y_train[i]
        
        # Add label text to image
        completed_steps = np.where(label == 1)[0]
        label_text = f"Steps completed: {completed_steps}"
        
        cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        output_path = os.path.join(output_dir, f"train_example_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return X_train, y_train, X_val, y_val

def preprocess_sensor_data(data_path, output_dir, window_size=100, step_size=20, val_split=0.2):
    """
    Preprocess sensor data for training
    
    Parameters:
    data_path (str): Path to the sensor data CSV file
    output_dir (str): Directory to save processed data
    window_size (int): Size of sliding window for time series
    step_size (int): Step size for sliding window
    val_split (float): Validation split ratio
    
    Returns:
    tuple: (X_train, y_train, X_val, y_val, sensor_columns)
    """
    print("Preprocessing sensor data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Sensor data file not found: {data_path}")
        return None, None, None, None, None
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded sensor data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        return None, None, None, None, None
    
    # Check required columns
    required_columns = ['pressure', 'temperature', 'proximity', 'vibration', 'step_id', 'completed']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None, None, None, None, None
    
    # Extract sensor columns
    sensor_columns = ['pressure', 'temperature', 'proximity', 'vibration']
    
    # Normalize sensor data
    scalers = {}
    for col in sensor_columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        scalers[col] = {'mean': mean, 'std': std}
    
    # Save scalers for inference
    with open(os.path.join(output_dir, 'sensor_scalers.json'), 'w') as f:
        json.dump(scalers, f)
    
    # Create sequences
    X = []
    y = []
    
    # Group by step_id to ensure sequences don't cross step boundaries
    print("Creating sequences from sensor data...")
    
    # Get unique step IDs
    step_ids = sorted(df['step_id'].unique())
    num_steps = len(step_ids)
    
    for step_id, group in tqdm(df.groupby('step_id')):
        sensor_data = group[sensor_columns].values
        completed = group['completed'].values
        
        for i in range(0, len(sensor_data) - window_size + 1, step_size):
            X.append(sensor_data[i:i+window_size])
            
            # Create a multi-hot vector where each position represents a step
            # A 1 means the step is completed
            label = np.zeros(num_steps)
            
            # Mark current step as completed based on the completed flag
            if completed[i+window_size-1]:
                label[step_id] = 1
            
            # Mark previous steps as completed (assuming sequential steps)
            for j in range(step_id):
                label[j] = 1
                
            y.append(label)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val, sensor_columns

def create_sensor_config(num_steps, sensor_columns):
    """
    Create a sensor configuration dictionary for SensorProcessor
    
    Parameters:
    num_steps (int): Number of steps in the process
    sensor_columns (list): List of sensor column names
    
    Returns:
    dict: Sensor configuration for SensorProcessor
    """
    # Create a simple sensor configuration
    sensor_config = {}
    for sensor in sensor_columns:
        sensor_config[sensor] = {}
    
    return sensor_config

def create_tf_datasets(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create TensorFlow datasets for training
    
    Parameters:
    X_train: Training features
    y_train: Training labels
    X_val: Validation features
    y_val: Validation labels
    batch_size: Batch size for training
    
    Returns:
    tuple: (train_dataset, val_dataset)
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset

def plot_training_history(history, model_name, output_dir):
    """
    Plot and save training history graphs
    
    Parameters:
    history: Training history object
    model_name (str): Name for saving the plots
    output_dir (str): Directory to save plots
    """
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_history.png'))
    
    # Save the history for later analysis
    with open(os.path.join(output_dir, f'{model_name}_history.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f)

def export_model_to_onnx(model, output_path, input_shape=None):
    """
    Export a TensorFlow model to ONNX format for Unity
    
    Parameters:
    model: TensorFlow model to export
    output_path (str): Path to save the ONNX model
    input_shape (tuple): Optional input shape override
    """
    try:
        import tf2onnx
        
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get input shape from model if not provided
        if input_shape is None:
            input_shape = model.input_shape
            
        # Handle batch dimension
        if input_shape[0] is None:
            input_shape = list(input_shape)
            input_shape[0] = 1
        
        # Define input signature
        input_signature = [tf.TensorSpec(input_shape, tf.float32, name='input')]
        
        # Convert the model to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        
        # Save the model
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        
        print(f"Model exported to {output_path}")
        return True
        
    except ImportError:
            print("Warning: tf2onnx not installed. Skipping ONNX export. Install with 'pip install tf2onnx'.")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train detection models')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--output', default='models', help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--train_vision', action='store_true', help='Train vision model')
    parser.add_argument('--train_sensor', action='store_true', help='Train sensor model')
    parser.add_argument('--export_unity', action='store_true', help='Export models for Unity')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune vision model')
    args = parser.parse_args()
    
    # If no training flags are set, enable both by default
    if not args.train_vision and not args.train_sensor:
        args.train_vision = True
        args.train_sensor = True
    
    # Load configuration
    config = load_config(args.config)
    
    # Get step definitions
    step_definitions = config["step_definitions"]
    num_steps = len(step_definitions)
    
    print(f"Process with {num_steps} steps: {step_definitions}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train vision model
    if args.train_vision:
        print("\n===== Training Vision Model =====\n")
        
        # Look for vision data in different possible locations
        vision_data_dir = None
        possible_locations = [
            os.path.join(args.dataset, 'vision'),
            args.dataset,
            os.path.join(args.dataset, 'vision', 'train')
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                # Check if this directory has step subdirectories
                has_steps = any(os.path.isdir(os.path.join(location, d)) and d.startswith('step_') for d in os.listdir(location))
                if has_steps:
                    vision_data_dir = location
                    break
        
        if vision_data_dir is None:
            print("Could not find vision data directory. Tried:", possible_locations)
            print("Available directories in dataset:", os.listdir(args.dataset))
        else:
            print(f"Using vision data from: {vision_data_dir}")
            
            # Preprocess vision data
            vision_processed_dir = os.path.join(args.output, 'vision_processed')
            X_train, y_train, X_val, y_val = preprocess_image_data(
                vision_data_dir, 
                vision_processed_dir, 
                input_shape=(224, 224)
            )
            
            if X_train is not None:
                if vision_processor_available:
                    print("Using VisionProcessor for model training")
                    
                    # Create model path
                    model_path = os.path.join(args.output, "vision_model.h5")
                    
                    # Create callbacks
                    callbacks = [
                        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
                        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, monitor='val_loss')
                    ]
                    
                    # Create and train the vision processor
                    vision_processor = VisionProcessor(num_process_steps=num_steps)
                    
                    # Train the model
                    history = vision_processor.train(
                        X_train, y_train,
                        X_val, y_val,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        callbacks=callbacks
                    )
                    
                    # Plot and save training history
                    plot_training_history(history, "vision_model", args.output)
                    
                    # Save the model
                    vision_processor.save_model(model_path)
                    print(f"Vision model saved to {model_path}")
                    
                    # Fine-tune if requested
                    if args.fine_tune:
                        print("\nFine-tuning vision model...")
                        
                        # Create fine-tuned model path
                        fine_tuned_model_path = os.path.join(args.output, "vision_model_fine_tuned.h5")
                        
                        # Create callbacks for fine-tuning
                        fine_tune_callbacks = [
                            tf.keras.callbacks.ModelCheckpoint(fine_tuned_model_path, save_best_only=True, monitor='val_loss'),
                            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7, monitor='val_loss')
                        ]
                        
                        # Fine-tune the model
                        ft_history = vision_processor.fine_tune(
                            X_train, y_train,
                            X_val, y_val,
                            learning_rate=args.learning_rate / 10,
                            epochs=10,
                            batch_size=args.batch_size,
                            callbacks=fine_tune_callbacks
                        )
                        
                        # Plot and save fine-tuning history
                        plot_training_history(ft_history, "vision_model_fine_tuned", args.output)
                        
                        # Save the fine-tuned model
                        vision_processor.save_model(fine_tuned_model_path)
                        print(f"Fine-tuned vision model saved to {fine_tuned_model_path}")
                        
                        # Use the fine-tuned model for export
                        model_for_export = vision_processor.model
                    else:
                        # Use the regular trained model for export
                        model_for_export = vision_processor.model
                else:
                    # Fall back to the original training code if VisionProcessor is not available
                    print("VisionProcessor not available, using standard model architecture")
                    
                    from tensorflow.keras.applications import MobileNetV2
                    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
                    from tensorflow.keras.models import Model
                    
                    # Build vision model
                    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                    base_model.trainable = False
                    
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(256, activation='relu')(x)
                    x = Dropout(0.5)(x)
                    outputs = Dense(num_steps, activation='sigmoid')(x)
                    
                    vision_model = Model(inputs=base_model.input, outputs=outputs)
                    vision_model.summary()
                    
                    # Create model path
                    model_path = os.path.join(args.output, "vision_model.h5")
                    
                    # Create callbacks
                    callbacks = [
                        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
                        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, monitor='val_loss')
                    ]
                    
                    # Compile the model
                    vision_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Train the model
                    history = vision_model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Plot and save training history
                    plot_training_history(history, "vision_model", args.output)
                    
                    # Use the trained model for export
                    model_for_export = vision_model
                
                # Export to ONNX if requested
                if args.export_unity:
                    unity_export_dir = os.path.join(args.output, 'unity_models')
                    os.makedirs(unity_export_dir, exist_ok=True)
                    export_model_to_onnx(
                        model_for_export,
                        os.path.join(unity_export_dir, 'vision_model.onnx')
                    )
    
    # Train sensor model
    if args.train_sensor:
        print("\n===== Training Sensor Model =====\n")
        
        # Look for sensor data file in different possible locations
        sensor_data_path = None
        possible_locations = [
            os.path.join(args.dataset, 'sensor_data.csv'),
            os.path.join(args.dataset, 'sensor', 'sensor_data.csv'),
            os.path.join(args.dataset, 'sensor', 'train_sensor_data.csv')
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                sensor_data_path = location
                break
        
        if sensor_data_path is None:
            print("Could not find sensor data file. Tried:", possible_locations)
            
            # Try to find any CSV file in the dataset directory
            csv_files = []
            for root, dirs, files in os.walk(args.dataset):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if csv_files:
                print(f"Found {len(csv_files)} CSV files in the dataset:")
                for file in csv_files:
                    print(f"  - {file}")
                
                # Use the first CSV file found
                sensor_data_path = csv_files[0]
                print(f"Using: {sensor_data_path}")
            else:
                print("No CSV files found in the dataset.")
        else:
            print(f"Using sensor data from: {sensor_data_path}")
        
        if sensor_data_path is not None:
            # Preprocess sensor data
            sensor_processed_dir = os.path.join(args.output, 'sensor_processed')
            window_size = 100  # 10 seconds at 10Hz
            step_size = 20  # 80% overlap
            
            X_train, y_train, X_val, y_val, sensor_columns = preprocess_sensor_data(
                sensor_data_path,
                sensor_processed_dir,
                window_size=window_size,
                step_size=step_size
            )
            
            if X_train is not None and sensor_columns is not None:
                # Create TensorFlow datasets
                train_dataset, val_dataset = create_tf_datasets(
                    X_train, y_train, X_val, y_val, batch_size=args.batch_size
                )
                
                if sensor_processor_available:
                    print("Using SensorProcessor for model training")
                    
                    # Create sensor configuration
                    sensor_config = create_sensor_config(num_steps, sensor_columns)
                    
                    # Create model path
                    model_path = os.path.join(args.output, "sensor_model.h5")
                    
                    # Create and train the sensor processor
                    sensor_processor = SensorProcessor(sensor_config)
                    
                    # Train the model
                    history = sensor_processor.train_model(
                        train_dataset,
                        val_dataset,
                        epochs=args.epochs
                    )
                    
                    # Plot and save training history
                    plot_training_history(history, "sensor_model", args.output)
                    
                    # Save the model
                    sensor_processor.analysis_model.save(model_path)
                    print(f"Sensor model saved to {model_path}")
                    
                    # Use the model for export
                    model_for_export = sensor_processor.analysis_model
                else:
                    # Fall back to standard approach if SensorProcessor is not available
                    print("SensorProcessor not available, using standard model architecture")
                    
                    from tensorflow.keras.layers import LSTM, Input, Dropout, Dense
                    from tensorflow.keras.models import Model
                    
                    # Get input shape
                    window_size, num_features = X_train.shape[1], X_train.shape[2]
                    sensor_input_shape = (window_size, num_features)
                    
                    # Build sensor model
                    inputs = Input(shape=sensor_input_shape)
                    x = LSTM(64, return_sequences=True)(inputs)
                    x = LSTM(32)(x)
                    x = Dense(16, activation='relu')(x)
                    x = Dropout(0.5)(x)
                    outputs = Dense(num_steps, activation='sigmoid')(x)
                    
                    sensor_model = Model(inputs=inputs, outputs=outputs)
                    sensor_model.summary()
                    
                    # Create model path
                    model_path = os.path.join(args.output, "sensor_model.h5")
                    
                    # Create callbacks
                    callbacks = [
                        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
                        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, monitor='val_loss')
                    ]
                    
                    # Compile the model
                    sensor_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Train the model
                    history = sensor_model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Plot and save training history
                    plot_training_history(history, "sensor_model", args.output)
                    
                    # Use the trained model for export
                    model_for_export = sensor_model
                
                # Export to ONNX if requested
                if args.export_unity:
                    unity_export_dir = os.path.join(args.output, 'unity_models')
                    os.makedirs(unity_export_dir, exist_ok=True)
                    export_model_to_onnx(
                        model_for_export,
                        os.path.join(unity_export_dir, 'sensor_model.onnx')
                    )
    
print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()