import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import cv2

# Import the base processor
from app_base_processor import BaseProcessor, DataProcessingError, PredictionError

class VisionProcessor(BaseProcessor):
    """Computer vision system for detecting completed tasks.
    
    Uses transfer learning with MobileNetV2 as a base model for efficient
    real-time processing on edge devices.
    """
    
    def __init__(self, num_process_steps, model_path=None, logger=None):
        """
        Initializes the vision system for task monitoring.
        
        Args:
            num_process_steps (int): Number of distinct steps in the task
            model_path (str, optional): Path to pre-trained model or weights
            logger (logging.Logger, optional): Logger instance
        """
        self.input_shape = (224, 224, 3)
        self.base_model = None
        
        # Initialize the parent class
        super().__init__(num_process_steps, model_path, logger)
    
    def _load_model(self, model_path):
        """
        Load model from path with specialized handling for vision models.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            tf.keras.Model: Loaded model
        """
        # Try to load as a complete model first
        model = super()._load_model(model_path)
        
        # Extract base model for reference
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                self.base_model = layer
                break
        
        # If we couldn't find a base model, it might be a custom architecture
        if self.base_model is None:
            self.base_model = MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
        
        return model
    
    def _build_model(self):
        """
        Build the vision model for step detection.
        
        Returns:
            tf.keras.Model: Vision model for step detection
        """
        # Create base model with ImageNet weights
        self.base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Feature extraction layers
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Custom classification head
        x = Dense(256, activation='relu', name='feature_dense_layer')(x)
        x = Dropout(0.5)(x)
        step_predictions = Dense(
            self.num_steps,
            activation='sigmoid',
            name='step_predictions')(x)
        
        model = Model(inputs=self.base_model.input, outputs=step_predictions)
        
        # Model will be compiled in BaseProcessor._prepare_for_training
        return model
    
    def preprocess_data(self, data):
        """
        Preprocess data for model input.
        
        This method handles both single images and batches of images.
        
        Args:
            data: Image data (NumPy array, path, or list of either)
            
        Returns:
            np.ndarray: Preprocessed data ready for model input
            
        Raises:
            DataProcessingError: If preprocessing fails
        """
        # If data is a string, assume it's a file path
        if isinstance(data, str):
            return self.preprocess_image_file(data)
        
        # If data is a list, process each item
        if isinstance(data, list):
            processed_images = []
            for item in data:
                if isinstance(item, str):
                    processed_images.append(self.preprocess_image_file(item))
                else:
                    processed_images.append(self.preprocess_frame(item))
            return np.vstack(processed_images)
        
        # Otherwise, assume it's an image array
        return self.preprocess_frame(data)
    
    def preprocess_image_file(self, image_path):
        """
        Preprocess an image file for model input.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Preprocessed image
            
        Raises:
            DataProcessingError: If image loading or preprocessing fails
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise DataProcessingError(f"Could not load image from {image_path}")
            
            return self.preprocess_frame(img)
        except Exception as e:
            raise DataProcessingError(f"Error preprocessing image file: {e}")
    
    def preprocess_frame(self, frame):
        """
        Prepares a camera frame for model input.
        
        Args:
            frame (np.array): Input BGR image from camera
            
        Returns:
            np.array: Preprocessed image ready for model prediction
            
        Raises:
            DataProcessingError: If preprocessing fails
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                raise DataProcessingError("Invalid frame: empty or None")
            
            # Check if we need to convert color
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed (OpenCV loads as BGR)
                if isinstance(frame, np.ndarray):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to expected dimensions
            resized_frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
            
            # Normalize pixel values
            normalized_frame = resized_frame / 255.0
            
            # Add batch dimension if needed
            if len(normalized_frame.shape) == 3:
                normalized_frame = np.expand_dims(normalized_frame, axis=0)
                
            return normalized_frame
        except cv2.error as e:
            raise DataProcessingError(f"OpenCV error during preprocessing: {e}")
        except Exception as e:
            raise DataProcessingError(f"Error preprocessing frame: {e}")
    
    def fine_tune(self, X_train, y_train, X_val=None, y_val=None, learning_rate=0.0001, 
                 epochs=10, batch_size=32, callbacks=None, unfreeze_layers=30):
        """
        Fine-tunes the model by unfreezing some of the base model layers
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            learning_rate (float): Lower learning rate for fine-tuning
            epochs (int): Number of fine-tuning epochs
            batch_size (int): Batch size for training
            callbacks (list): Optional callbacks for training
            unfreeze_layers (int): Number of layers to unfreeze from the end
            
        Returns:
            tf.keras.History: Training history metrics
        """
        if callbacks is None:
            callbacks = []
            
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers for fine-tuning
        for i, layer in enumerate(self.base_model.layers):
            # Unfreeze only the last 'unfreeze_layers' layers
            layer.trainable = i >= (len(self.base_model.layers) - unfreeze_layers)
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Determine validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Fine-tune the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_weights(self, filepath):
        """
        Save just the model weights to a file
        
        Args:
            filepath (str): Path to save the weights
        """
        if self.model is None:
            raise ValueError("No model to save weights from")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        self.model.save_weights(filepath)
        self.logger.info(f"Model weights saved to {filepath}")
    
    def _prepare_for_training(self, optimizer):
        """
        Prepare the model for training by freezing the base model layers.
        
        Args:
            optimizer: The optimizer to use for training
        """
        # Freeze the base model for transfer learning
        self.base_model.trainable = False
        
        # Compile the model with the provided optimizer
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )