import numpy as np
import pandas as pd
from scipy import signal
import json
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from app_base_processor import BaseProcessor, DataProcessingError, PredictionError

class SensorProcessor(BaseProcessor):
    """Analyzes sensor data to detect tasks completion.
    
    Uses signal processing and LSTM neural networks to identify patterns
    in time-series sensor data that correspond to specific assembly steps.
    """

    def __init__(self, sensor_config, model_path=None, logger=None, window_size=100):
        """
        Initializes the sensor data analysis system.
        
        Args:
            sensor_config (dict): Configuration mapping sensor names to their 
                                expected signal patterns for each process step
            model_path (str, optional): Path to pre-trained model weights
            logger (logging.Logger, optional): Logger instance
            window_size (int): Window size for time series analysis
        """
        self.sensor_config = sensor_config or {}
        self.sensor_names = list(self.sensor_config.keys())
        
        # If sensor_names is empty, use default sensor names
        if not self.sensor_names:
            self.sensor_names = ['pressure', 'temperature', 'proximity', 'vibration']
            if logger:
                logger.warning(f"No sensor names found in config, using defaults: {self.sensor_names}")
        
        self.window_size = window_size
        
        # Create dictionary for sensor scalers
        self.scalers = {sensor: {'mean': 0, 'std': 1} for sensor in self.sensor_names}
        
        # Determine number of steps from sensor config or use default
        num_steps = 5  # Default to 5 steps
        
        # Initialize the parent class
        super().__init__(num_steps, model_path, logger)

    def _build_model(self):
        """
        Constructs an LSTM neural network for analyzing sensor time series data.
        
        Returns:
            tf.keras.Model: Compiled LSTM model for step detection
        """
        model = Sequential([
            LSTM(64, return_sequences=True, 
                input_shape=(self.window_size, len(self.sensor_names)),
                name='lstm_feature_extractor'),
            LSTM(32, name='lstm_context_analyzer'),
            Dense(16, activation='relu', name='feature_processor'),
            Dropout(0.5),
            Dense(self.num_steps, activation='sigmoid', name='step_predictions')
        ])
        
        # Model will be compiled in BaseProcessor._prepare_for_training
        return model
    
    def load_scalers(self, filepath):
        """
        Load scalers from file.
        
        Args:
            filepath (str): Path to load scalers from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                self.scalers = json.load(f)
            self.logger.info(f"Loaded scalers from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
            return False
    
    def save_scalers(self, filepath):
        """
        Save scalers to file.
        
        Args:
            filepath (str): Path to save scalers
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.scalers, f, indent=2)
            self.logger.info(f"Saved scalers to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving scalers: {e}")
            return False
    
    def update_scalers(self, data):
        """
        Update scalers with new data statistics.
        
        Args:
            data (pd.DataFrame): Sensor data
        """
        for sensor in self.sensor_names:
            if sensor in data.columns:
                self.scalers[sensor]['mean'] = float(data[sensor].mean())
                self.scalers[sensor]['std'] = float(data[sensor].std()) or 1.0  # Avoid division by zero
    
    def apply_scalers(self, data):
        """
        Apply scaling to sensor data.
        
        Args:
            data (pd.DataFrame): Sensor data
            
        Returns:
            pd.DataFrame: Scaled data
        """
        scaled_data = data.copy()
        for sensor in self.sensor_names:
            if sensor in data.columns:
                mean = self.scalers[sensor]['mean']
                std = self.scalers[sensor]['std']
                scaled_data[sensor] = (data[sensor] - mean) / std
        return scaled_data
    
    def apply_filters(self, data):
        """
        Apply bandpass filters to sensor data.
        
        Args:
            data (pd.DataFrame): Sensor data
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = pd.DataFrame()
        for sensor in self.sensor_names:
            if sensor in data.columns:
                # Apply a bandpass filter to remove noise
                b, a = signal.butter(3, [0.05, 0.95], 'bandpass')
                filtered_data[sensor] = signal.filtfilt(b, a, data[sensor])
        return filtered_data
    
    def preprocess_data(self, data):
        """
        Preprocess data for model input.
        
        Args:
            data: Sensor data (DataFrame, CSV path, or array)
            
        Returns:
            np.ndarray: Preprocessed data ready for model input
            
        Raises:
            DataProcessingError: If preprocessing fails
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, str):
                # Assume it's a CSV file path
                if not os.path.exists(data):
                    raise DataProcessingError(f"File not found: {data}")
                data = pd.read_csv(data)
            elif isinstance(data, np.ndarray):
                # Convert array to DataFrame
                if data.shape[1] != len(self.sensor_names):
                    raise DataProcessingError(
                        f"Array shape mismatch: expected {len(self.sensor_names)} "
                        f"columns but got {data.shape[1]}"
                    )
                data = pd.DataFrame(data, columns=self.sensor_names)
            
            # Validate required columns
            missing_columns = [col for col in self.sensor_names if col not in data.columns]
            if missing_columns:
                raise DataProcessingError(f"Missing required columns: {missing_columns}")
            
            # Apply filters
            filtered_data = self.apply_filters(data)
            
            # Apply scaling
            scaled_data = self.apply_scalers(filtered_data)
            
            # Extract relevant columns
            sensor_data = scaled_data[self.sensor_names].values
            
            # Create windows
            if len(sensor_data) < self.window_size:
                # Pad data if not enough samples
                pad_rows = self.window_size - len(sensor_data)
                padding = np.zeros((pad_rows, len(self.sensor_names)))
                sensor_data = np.vstack([padding, sensor_data])
            
            # Reshape for LSTM input: [samples, time_steps, features]
            # Create overlapping windows if there's enough data
            windows = []
            if len(sensor_data) > self.window_size:
                for i in range(len(sensor_data) - self.window_size + 1):
                    window = sensor_data[i:i + self.window_size]
                    windows.append(window)
                processed_data = np.array(windows)
            else:
                # Just use the padded data as a single window
                processed_data = np.expand_dims(sensor_data, axis=0)
            
            return processed_data
        except DataProcessingError:
            # Re-raise preprocessing errors
            raise
        except Exception as e:
            raise DataProcessingError(f"Error preprocessing sensor data: {e}")
    
    def create_dataset_from_raw_data(self, data, step_column='step_id', 
                                    completion_column='completed', 
                                    window_size=None, step_size=20):
        """
        Creates a training dataset from raw sensor data.
        
        Args:
            data (pd.DataFrame): Raw sensor data
            step_column (str): Column name for step ID
            completion_column (str): Column name for completion status
            window_size (int): Size of sliding window for time series
            step_size (int): Step size for sliding window
            
        Returns:
            tuple: (X, y) for training
        """
        if window_size is None:
            window_size = self.window_size
        
        # Extract sensor columns
        sensor_data = data[self.sensor_names].values
        step_ids = data[step_column].values
        completion = data[completion_column].values
        
        # Create sequences
        X = []
        y = []
        
        # Group by step_id to ensure sequences don't cross step boundaries
        for step_id in np.unique(step_ids):
            # Get indices for this step
            indices = np.where(step_ids == step_id)[0]
            if len(indices) < window_size:
                continue
            
            # Get data for this step
            step_sensor_data = sensor_data[indices]
            step_completion = completion[indices]
            
            # Create sliding windows
            for i in range(0, len(step_sensor_data) - window_size + 1, step_size):
                window = step_sensor_data[i:i+window_size]
                X.append(window)
                
                # Create a multi-hot vector where each position represents a step
                # A 1 means the step is completed
                label = np.zeros(self.num_steps)
                
                # Mark current step as completed based on the completion flag
                if step_completion[i+window_size-1]:
                    label[step_id] = 1
                
                # Mark previous steps as completed (assuming sequential steps)
                for j in range(step_id):
                    label[j] = 1
                    
                y.append(label)
        
        return np.array(X), np.array(y)