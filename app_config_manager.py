import os
import sys
import json
from datetime import datetime

class ConfigManager:
    """Manages configuration for process monitoring."""
  
    def __init__(self, config_path=None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Define a comprehensive default configuration
        self.config = {
            'system': {
                'name': 'ML Monitoring',
                'version': '1.0.0'
            },
            'step_definitions': [],
            'sensor_config': {},
            'model_paths': {
                'vision': None,
                'sensor': None,
                'sensor_scalers': None
            },
            'camera_sources': [],
            'training': {
                'batch_size': 32,
                'epochs': 30,
                'learning_rate': 0.001,
                'validation_split': 0.2
            },
            'inference': {
                'confidence_threshold': 0.5,
                'window_size': 100
            }
        }
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from file.
        
        Args:
            config_path (str): Path to configuration file
        
        Raises:
            FileNotFoundError: If config file does not exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self._merge_config(loaded_config)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file: {config_path}")
            raise
    
    def _merge_config(self, new_config):
        """
        Recursively merge new configuration with existing configuration.
        
        Args:
            new_config (dict): New configuration to merge
        """
        def _deep_merge(original, update):
            """
            Recursively merge two dictionaries.
            
            Args:
                original (dict): Original dictionary
                update (dict): Dictionary to merge with original
            
            Returns:
                dict: Merged dictionary
            """
            for key, value in update.items():
                if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                    original[key] = _deep_merge(original[key], value)
                else:
                    original[key] = value
            return original
        
        self.config = _deep_merge(self.config, new_config)
    
    def get_config(self):
        """
        Get complete configuration.
        
        Returns:
            dict: Complete configuration
        """
        return self.config
    
    def get_value(self, key_path, default=None):
        """
        Get configuration value using dot notation path.
        
        Args:
            key_path (str): Dot-separated path to configuration value
            default (Any, optional): Default value if key is not found
        
        Examples:
            config.get_value('training.batch_size', 32)
            config.get_value('model_paths.vision')
        
        Returns:
            Value of the configuration key or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key_path, value):
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to configuration value
            value (Any): Value to set
        
        Example:
            config.set_value('training.batch_size', 64)
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to the parent of the last key
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the final key
        current[keys[-1]] = value
    
    def save_config(self, config_path):
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path (str): Path to save configuration file
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")