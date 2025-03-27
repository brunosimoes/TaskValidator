import os
import logging
import tensorflow as tf
import numpy as np

class ProcessorError(Exception):
    """Base exception for processor errors."""
    pass

class ModelLoadingError(ProcessorError):
    """Exception raised when model loading fails."""
    pass

class DataProcessingError(ProcessorError):
    """Exception raised when data processing fails."""
    pass

class PredictionError(ProcessorError):
    """Exception raised when prediction fails."""
    pass

class BaseProcessor:
    """Base class for process step detection processors."""
    
    def __init__(self, num_steps, model_path=None, logger=None):
        """
        Initialize the base processor.
        
        Args:
            num_steps (int): Number of steps to detect
            model_path (str, optional): Path to pre-trained model
            logger (logging.Logger, optional): Logger instance
        """
        self.num_steps = num_steps
        self.model_path = model_path
        self.model = None
        
        # Set up logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load or build model
        self._load_or_create_model(model_path)
    
    def _load_or_create_model(self, model_path):
        """
        Load model from path or create a new one.
        
        Args:
            model_path (str, optional): Path to pre-trained model
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        if model_path and os.path.exists(model_path):
            try:
                self.model = self._load_model(model_path)
                self.logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Could not load model: {e}")
                self.logger.info("Building new model")
                self.model = self._build_model()
        else:
            self.logger.info("Building new model")
            self.model = self._build_model()
    
    def _load_model(self, model_path):
        """
        Load model from path.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            tf.keras.Model: Loaded model
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def _build_model(self):
        """
        Build model architecture.
        
        Returns:
            tf.keras.Model: Built model
            
        Raises:
            NotImplementedError: Method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _build_model")
    
    def _prepare_for_training(self, optimizer):
        """
        Prepare the model for training. This method is called by the train method
        before model.fit is called.
        
        Args:
            optimizer: The optimizer to use for training
            
        Raises:
            NotImplementedError: Method may be implemented by subclasses
        """
        # Default implementation - just compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, 
              batch_size=32, callbacks=None, learning_rate=0.001):
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): Optional callbacks for training
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            tf.keras.History: Training history
        """
        if callbacks is None:
            callbacks = []
        
        # Create optimizer with specified learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Prepare the model for training (subclasses may override this)
        self._prepare_for_training(optimizer)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, input_data, verbose=0):
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data to predict on
            
        Returns:
            numpy.ndarray: Predictions
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            if self.model is None:
                raise PredictionError("No model available for prediction")
            
            # Preprocess input data
            processed_data = self.preprocess_data(input_data)
            
            # Make predictions
            return self.model.predict(processed_data, verbose=verbose)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}")
    
    def preprocess_data(self, data):
        """
        Preprocess data for model input.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data
            
        Raises:
            NotImplementedError: Method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement preprocess_data")
    
    def detect_steps(self, input_data, confidence_threshold=0.5):
        """
        Identifies completed assembly steps from input data.
        
        Args:
            input_data: Input data for prediction
            confidence_threshold (float): Threshold for determining step completion
            
        Returns:
            dict: Contains:
                - predictions: Confidence scores for each process step (0-1)
                - completed_steps: List of step IDs with confidence > threshold
                - all_completed: Boolean indicating full process completion
                
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Get predictions
            predictions = self.predict(input_data)
            
            # Handle batch vs single predictions
            if len(predictions.shape) > 1 and predictions.shape[0] > 1:
                # Handle batch predictions by averaging
                mean_predictions = np.mean(predictions, axis=0)
            else:
                # Extract single prediction
                mean_predictions = predictions[0]
            
            # Determine completed steps using confidence threshold
            completed_steps = [
                step_id 
                for step_id, confidence in enumerate(mean_predictions) 
                if confidence > confidence_threshold
            ]
            
            return {
                'predictions': mean_predictions,
                'completed_steps': completed_steps,
                'all_completed': len(completed_steps) == self.num_steps
            }
        except Exception as e:
            raise PredictionError(f"Error during step detection: {e}")
    
    def save_model(self, filepath):
        """
        Save the model.
        
        Args:
            filepath (str): Path to save the model
            
        Raises:
            ValueError: If no model is available to save
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def evaluate(self, X_test, y_test, verbose=0):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            ValueError: If no model is available for evaluation
        """
        if self.model is None:
            raise ValueError("No model to evaluate")
        
        # Preprocess test data if needed
        processed_X_test = X_test
        if hasattr(self, 'preprocess_data'):
            processed_X_test = self.preprocess_data(X_test)
        
        # Calculate loss and accuracy
        loss, accuracy = self.model.evaluate(processed_X_test, y_test, verbose=verbose)
        
        # Get predictions
        predictions = self.model.predict(processed_X_test, verbose=verbose)
        
        # Calculate per-step metrics
        per_step_metrics = self._calculate_per_step_metrics(predictions, y_test)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'per_step_metrics': per_step_metrics
        }
        
    def _calculate_per_step_metrics(self, predictions, y_test, threshold=0.5):
        """
        Calculate metrics for each step.
        
        Args:
            predictions (np.ndarray): Model predictions
            y_test (np.ndarray): Ground truth labels
            threshold (float): Threshold for binary classification
            
        Returns:
            list: List of metrics for each step
        """
        # Apply threshold
        binary_preds = predictions > threshold
        binary_truth = y_test > threshold
        
        # Calculate metrics for each step
        per_step_metrics = []
        for step_idx in range(y_test.shape[1]):
            step_preds = binary_preds[:, step_idx]
            step_truth = binary_truth[:, step_idx]
            
            # Calculate TP, FP, TN, FN
            tp = np.sum((step_preds == True) & (step_truth == True))
            fp = np.sum((step_preds == True) & (step_truth == False))
            tn = np.sum((step_preds == False) & (step_truth == False))
            fn = np.sum((step_preds == False) & (step_truth == True))
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_step_metrics.append({
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            })
        
        return per_step_metrics
    
    def get_model_summary(self):
        """
        Get a summary of the model architecture.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model available"
        
        # Create a string buffer to capture the summary
        summary_lines = []
        self.model.summary(print_fn=lambda line: summary_lines.append(line))
        
        return "\n".join(summary_lines)