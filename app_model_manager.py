import os
import json
import tensorflow as tf
import numpy as np
from datetime import datetime

from app_vision_processor import VisionProcessor
from app_sensor_processor import SensorProcessor
from app_data_utils import MetricsCalculator, TrainingHistoryPlotter

class ModelManager:
    """
    Manages model training, evaluation, and export.
    Works with both VisionProcessor and SensorProcessor.
    """
    
    def __init__(self, config_manager, logger):
        """
        Initialize model manager.
        
        Args:
            config_manager (ConfigManager): Configuration manager
            logger (Logger): Logger instance
        """
        self.config = config_manager
        self.logger = logger
        self.processors = {}
    
    def get_processor(self, processor_type, reload=False):
        """
        Get processor instance by type.
        
        Args:
            processor_type (str): 'vision' or 'sensor'
            reload (bool): Force reload of processor
            
        Returns:
            BaseProcessor: Processor instance
        """
        if processor_type in self.processors and not reload:
            return self.processors[processor_type]
        
        if processor_type == 'vision':
            # Get number of steps from config
            num_steps = len(self.config.get_value('step_definitions', []))
            
            # If num_steps is 0, use a default value of 5 (based on your data)
            if num_steps == 0:
                self.logger.warning("No step definitions found in config, defaulting to 5 steps")
                num_steps = 5
                
            model_path = self.config.get_value('model_paths.vision')
            
            processor = VisionProcessor(num_steps, model_path, self.logger)
        elif processor_type == 'sensor':
            sensor_config = self.config.get_value('sensor_config', {})
            model_path = self.config.get_value('model_paths.sensor')
            
            processor = SensorProcessor(sensor_config, model_path, self.logger)
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        self.processors[processor_type] = processor
        return processor
        
    def train_model(self, processor_type, train_data, val_data=None, output_dir=None):
        """
        Train a model using the specified processor.
        
        Args:
            processor_type (str): 'vision' or 'sensor'
            train_data (dict or tuple): Training data dictionary or (X_train, y_train) tuple
            val_data (dict or tuple, optional): Validation data
            output_dir (str, optional): Directory to save model and results
            
        Returns:
            dict: Training results
        """
        # Get processor
        processor = self.get_processor(processor_type)
        
        # Extract X_train and y_train from train_data
        if isinstance(train_data, dict):
            X_train = train_data['X_train']
            y_train = train_data['y_train']
            
            # Extract validation data if provided
            X_val = train_data.get('X_val')
            y_val = train_data.get('y_val')
        elif isinstance(train_data, tuple) and len(train_data) == 2:
            X_train, y_train = train_data
            
            # Extract validation data if provided
            if val_data and isinstance(val_data, tuple) and len(val_data) == 2:
                X_val, y_val = val_data
            else:
                X_val, y_val = None, None
        else:
            raise TypeError("train_data must be a dictionary with 'X_train' and 'y_train' keys or a tuple of (X_train, y_train)")
        
        # Make sure number of steps matches the data
        if processor_type == 'vision':
            data_num_steps = y_train.shape[1]
            
            if processor.num_steps != data_num_steps:
                self.logger.warning(f"Model has {processor.num_steps} steps but data has {data_num_steps} steps")
                self.logger.info(f"Recreating processor with {data_num_steps} steps")
                
                # Update config with proper step count
                if len(self.config.get_value('step_definitions', [])) != data_num_steps:
                    # Create default step definitions if needed
                    self.config.config['step_definitions'] = [f"Step {i}" for i in range(data_num_steps)]
                
                # Create new processor with correct number of steps
                processor = VisionProcessor(data_num_steps, None, self.logger)
                self.processors[processor_type] = processor

        # Make sure sensor_columns are correctly set for sensor models
        if processor_type == 'sensor' and isinstance(processor, SensorProcessor):
            # Update sensor_names if needed
            if isinstance(train_data, dict) and 'sensor_columns' in train_data:
                processor.sensor_names = train_data['sensor_columns']
                processor.logger.info(f"Updated sensor names to: {processor.sensor_names}")
            
            # Rebuild model if sensor_names don't match
            if len(processor.sensor_names) != X_train.shape[2]:
                processor.logger.warning(f"Model has {len(processor.sensor_names)} sensors but data has {X_train.shape[2]} sensors")
                processor.logger.info(f"Rebuilding sensor processor for {X_train.shape[2]} sensors")
                
                # Update sensor_names based on data
                if isinstance(train_data, dict) and 'sensor_columns' in train_data:
                    processor.sensor_names = train_data['sensor_columns']
                else:
                    processor.sensor_names = [f"sensor_{i}" for i in range(X_train.shape[2])]
                
                # Rebuild the model
                processor.model = processor._build_model()

        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up callbacks
        model_path = os.path.join(output_dir, f"{processor_type}_model.h5") if output_dir else None
        callbacks = []
        
        if model_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    model_path, save_best_only=True, monitor='val_loss'
                )
            )
        
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                patience=10, monitor='val_loss', restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.1, patience=5, min_lr=1e-6, monitor='val_loss'
            )
        ])
        
        # Get training parameters
        epochs = self.config.get_value('training.epochs', 30)
        batch_size = self.config.get_value('training.batch_size', 32)
        learning_rate = self.config.get_value('training.learning_rate', 0.001)
        
        # Train the model
        self.logger.info(f"Starting {processor_type} model training...")
        
        # Train with validation data if available
        if X_val is not None and y_val is not None:
            history = processor.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                learning_rate=learning_rate
            )
        else:
            history = processor.train(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                learning_rate=learning_rate
            )
        
        # Save the model
        if model_path:
            processor.save_model(model_path)
            self.logger.info(f"{processor_type.capitalize()} model saved to {model_path}")
            
            # Save training history plot
            if output_dir and hasattr(history, 'history'):
                TrainingHistoryPlotter.plot_training_history(
                    history, 
                    f"{processor_type.capitalize()} Model", 
                    output_dir
                )
        
        return {
            'model_path': model_path,
            'history': history.history if hasattr(history, 'history') else None,
            'output_dir': output_dir
        }
    
    def evaluate_model(self, processor_type, test_data, output_dir=None):
        """
        Evaluate a model on test data.
        
        Args:
            processor_type (str): 'vision' or 'sensor'
            test_data (tuple): Test data (X_test, y_test)
            output_dir (str, optional): Directory to save evaluation results
            
        Returns:
            dict: Evaluation results
        """
        processor = self.get_processor(processor_type)
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"evaluation/{processor_type}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
        
        # Unpack test data
        X_test, y_test = test_data
        
        # Evaluate model
        self.logger.info(f"Evaluating {processor_type} model...")
        
        # Use the processor's evaluate method (now in BaseProcessor)
        eval_metrics = processor.evaluate(X_test, y_test)
        
        # Save evaluation results
        with open(os.path.join(output_dir, f'{processor_type}_evaluation.json'), 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Create confusion matrix plots using MetricsCalculator
        predictions = processor.predict(X_test)
        
        # Get step definitions for plot labels
        step_definitions = self.config.get_value('step_definitions', [])
        step_names = [f"Step {i+1}: {step}" for i, step in enumerate(step_definitions)]
        if len(step_names) != y_test.shape[1]:
            step_names = [f"Step {i+1}" for i in range(y_test.shape[1])]
        
        # Create confusion matrix plot
        MetricsCalculator.plot_confusion_matrices(
            predictions, 
            y_test, 
            threshold=self.config.get_value('inference.confidence_threshold', 0.5),
            step_names=step_names,
            save_path=os.path.join(output_dir, f'{processor_type}_confusion_matrix.png')
        )
        
        return {
            'metrics': eval_metrics,
            'output_dir': output_dir
        }