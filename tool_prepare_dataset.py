import os
import shutil
import random
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_image_data(input_dir, output_dir, test_size=0.2, val_size=0.1):
    """
    Split image data into train, validation, and test sets
    
    Parameters:
    input_dir (str): Directory containing generated images
    output_dir (str): Directory to save the split datasets
    test_size (float): Proportion of data for testing (0-1)
    val_size (float): Proportion of data for validation (0-1)
    """
    print("Splitting image data into train/val/test sets...")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each step directory
    for step_dir in os.listdir(input_dir):
        step_path = os.path.join(input_dir, step_dir)
        if not os.path.isdir(step_path):
            continue
        
        print(f"Processing {step_dir}...")
        
        # Create step directories in train/val/test
        train_step_dir = os.path.join(train_dir, step_dir)
        val_step_dir = os.path.join(val_dir, step_dir)
        test_step_dir = os.path.join(test_dir, step_dir)
        
        os.makedirs(train_step_dir, exist_ok=True)
        os.makedirs(val_step_dir, exist_ok=True)
        os.makedirs(test_step_dir, exist_ok=True)
        
        # Process complete and incomplete subdirectories
        for status_dir in ['complete', 'incomplete']:
            status_path = os.path.join(step_path, status_dir)
            if not os.path.isdir(status_path):
                continue
            
            # Create status directories in train/val/test
            train_status_dir = os.path.join(train_step_dir, status_dir)
            val_status_dir = os.path.join(val_step_dir, status_dir)
            test_status_dir = os.path.join(test_step_dir, status_dir)
            
            os.makedirs(train_status_dir, exist_ok=True)
            os.makedirs(val_status_dir, exist_ok=True)
            os.makedirs(test_status_dir, exist_ok=True)
            
            # Get all images
            image_files = [f for f in os.listdir(status_path) if f.endswith('.jpg')]
            
            # Split into train, validation, and test
            train_val_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
            train_files, val_files = train_test_split(train_val_files, test_size=val_size/(1-test_size), random_state=42)
            
            # Copy files to their respective directories
            for file in tqdm(train_files, desc=f"Copying train/{status_dir}"):
                src = os.path.join(status_path, file)
                dst = os.path.join(train_status_dir, file)
                shutil.copy2(src, dst)
            
            for file in tqdm(val_files, desc=f"Copying val/{status_dir}"):
                src = os.path.join(status_path, file)
                dst = os.path.join(val_status_dir, file)
                shutil.copy2(src, dst)
            
            for file in tqdm(test_files, desc=f"Copying test/{status_dir}"):
                src = os.path.join(status_path, file)
                dst = os.path.join(test_status_dir, file)
                shutil.copy2(src, dst)

def split_sensor_data(input_file, output_dir, test_size=0.2, val_size=0.1):
    """
    Split sensor data into train, validation, and test sets
    
    Parameters:
    input_file (str): CSV file containing generated sensor data
    output_dir (str): Directory to save the split datasets
    test_size (float): Proportion of data for testing (0-1)
    val_size (float): Proportion of data for validation (0-1)
    """
    print("Splitting sensor data into train/val/test sets...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sensor data
    df = pd.read_csv(input_file)
    
    # Split by timestamp to maintain sequential nature
    timestamps = df['timestamp'].unique()
    
    # Split timestamps into train, validation, and test
    train_val_timestamps, test_timestamps = train_test_split(timestamps, test_size=test_size, random_state=42)
    train_timestamps, val_timestamps = train_test_split(train_val_timestamps, test_size=val_size/(1-test_size), random_state=42)
    
    # Create dataframes for each split
    train_df = df[df['timestamp'].isin(train_timestamps)]
    val_df = df[df['timestamp'].isin(val_timestamps)]
    test_df = df[df['timestamp'].isin(test_timestamps)]
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, 'train_sensor_data.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_sensor_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_sensor_data.csv'), index=False)
    
    print(f"Sensor data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test samples")

def create_metadata(output_dir, config):
    """
    Create metadata files for the dataset
    
    Parameters:
    output_dir (str): Output directory
    config (dict): Configuration with step definitions and sensor info
    """
    print("Creating metadata files...")
    
    metadata = {
        "dataset_name": "ML Dataset",
        "version": "1.0",
        "step_definitions": config["step_definitions"],
        "sensor_config": config["sensor_config"],
        "train_test_split": {
            "train": 0.7,
            "validation": 0.1,
            "test": 0.2
        },
        "image_format": "JPEG",
        "image_dimensions": "640x480"
    }
    
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input_dir', required=True, help='Input directory with generated dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory for split dataset')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data for validation')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        import json
        config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split image data
    image_input_dir = os.path.join(args.input_dir, 'vision')
    image_output_dir = os.path.join(args.output_dir, 'vision')
    split_image_data(image_input_dir, image_output_dir, args.test_size, args.val_size)
    
    # Split sensor data
    sensor_input_file = os.path.join(args.input_dir, 'sensor_data.csv')
    sensor_output_dir = os.path.join(args.output_dir, 'sensor')
    split_sensor_data(sensor_input_file, sensor_output_dir, args.test_size, args.val_size)
    
    # Create metadata
    create_metadata(args.output_dir, config)
    
    print("Dataset splitting complete!")

if __name__ == '__main__':
    main()