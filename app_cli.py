import argparse
import os
import sys
import json
from datetime import datetime
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AssemblyML:
    """Command-line interface for the tool."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        self.args = None
        
        # Initialize logger
        from app_logger import Logger
        self.logger = Logger('task-validator-cli')
        
        # Initialize config manager
        from app_config_manager import ConfigManager
        self.config_manager = None
    
    def _create_parser(self):
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description='Task ML Tools',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  # Generate synthetic dataset
  python app_cli.py generate --config config.json --output dataset
  
  # Prepare dataset for training
  python app_cli.py prepare --config config.json --input dataset --output prepared_data
  
  # Train models
  python app_cli.py train --data prepared_data --config config.json
  
  # Test models
  python app_cli.py test --config config.json --vision-model models/vision/vision_model.h5 --image dataset/vision/step_1_Circuit_Board_Mounting/complete/0071.jpg
  python app_cli.py test --config config.json --vision-model models/vision/vision_model.h5 --sensor-model models/sensor/sensor_model.h5 --image dataset/vision/step_3_Cable_Connection/complete/0020.jpg --sensor-data prepared_data/sensor/test_sensor_data.csv

  # Run monitoring application
  python app_cli.py run --config config.json
''')
        
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Generate command
        generate_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
        generate_parser.add_argument('--config', required=True, help='Path to configuration file')
        generate_parser.add_argument('--output', required=True, help='Output directory for dataset')
        generate_parser.add_argument('--num-sensor-samples', type=int, default=10000, 
                                    help='Number of sensor data samples')
        generate_parser.add_argument('--num-images-per-step', type=int, default=100, 
                                    help='Number of images per step')
        
        # Prepare command
        prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset for training')
        prepare_parser.add_argument('--input', required=True, help='Input directory with generated dataset')
        prepare_parser.add_argument('--output', required=True, help='Output directory for prepared dataset')
        prepare_parser.add_argument('--config', required=True, help='Path to configuration file')
        prepare_parser.add_argument('--test-size', type=float, default=0.2, 
                                   help='Proportion of data for testing')
        prepare_parser.add_argument('--val-size', type=float, default=0.1, 
                                   help='Proportion of data for validation')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train models')
        train_parser.add_argument('--data', required=True, help='Path to prepared dataset')
        train_parser.add_argument('--config', required=True, help='Path to configuration file')
        train_parser.add_argument('--output', default='models', help='Path to output directory')
        train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
        train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
        train_parser.add_argument('--learning-rate', type=float, default=0.001, 
                                 help='Learning rate for optimizer')
        train_parser.add_argument('--vision', action='store_true', help='Train vision model')
        train_parser.add_argument('--sensor', action='store_true', help='Train sensor model')
        train_parser.add_argument('--export-unity', action='store_true', help='Export models for Unity')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test models')
        test_parser.add_argument('--config', required=True, help='Path to configuration file')
        test_parser.add_argument('--vision-model', help='Path to trained vision model')
        test_parser.add_argument('--sensor-model', help='Path to trained sensor model')
        test_parser.add_argument('--image', help='Path to test image')
        test_parser.add_argument('--sensor-data', help='Path to test sensor data CSV file')
        test_parser.add_argument('--scaler', help='Path to sensor scaler JSON file')
        test_parser.add_argument('--output', default='test_results', help='Path to save test results')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Run monitoring application')
        run_parser.add_argument('--config', required=True, help='Path to configuration file')
        run_parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
        run_parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
        run_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
        
        return parser
    
    def run(self):
        """Run the CLI application."""
        self.args = self.parser.parse_args()
        
        if not self.args.command:
            self.parser.print_help()
            return
        
        # Load configuration if provided
        if hasattr(self.args, 'config') and self.args.config:
            from app_config_manager import ConfigManager
            self.config_manager = ConfigManager(self.args.config)
            self.logger.info(f"Loaded configuration from {self.args.config}")
        
        # Execute command
        command_method = f"_cmd_{self.args.command}"
        if hasattr(self, command_method):
            getattr(self, command_method)()
        else:
            self.logger.error(f"Unknown command: {self.args.command}")
    
    def _cmd_generate(self):
        """Execute generate command."""
        self.logger.info("Generating synthetic dataset...")
        
        # Import generation module
        from tool_generate_dataset import main as generate_main
        
        # Set up command-line arguments for generate_main
        sys.argv = [
            sys.argv[0],
            '--config', self.args.config,
            '--output_dir', self.args.output,
            '--num_sensor_samples', str(self.args.num_sensor_samples),
            '--num_images_per_step', str(self.args.num_images_per_step)
        ]
        
        # Run generate_main
        generate_main()
        
        self.logger.info(f"Dataset generation complete! Output directory: {self.args.output}")
    
    def _cmd_prepare(self):
        """Execute prepare command."""
        self.logger.info("Preparing dataset for training...")
        
        # Import preparation module
        from tool_prepare_dataset import main as prepare_main
        
        # Set up command-line arguments for prepare_main
        sys.argv = [
            sys.argv[0],
            '--input_dir', self.args.input,
            '--output_dir', self.args.output,
            '--config', self.args.config,
            '--test_size', str(self.args.test_size),
            '--val_size', str(self.args.val_size)
        ]
        
        # Run prepare_main
        prepare_main()
        
        self.logger.info(f"Dataset preparation complete! Output directory: {self.args.output}")
    
    def _cmd_train(self):
        """Execute train command."""
        self.logger.info("Training models...")
        
        # Create model manager
        from app_model_manager import ModelManager
        model_manager = ModelManager(self.config_manager, self.logger)
        
        # Update configuration with command-line arguments
        self.config_manager.config['training']['batch_size'] = self.args.batch_size
        self.config_manager.config['training']['epochs'] = self.args.epochs
        self.config_manager.config['training']['learning_rate'] = self.args.learning_rate
        
        # Train vision model if requested
        if self.args.vision or not (self.args.vision or self.args.sensor):
            self.logger.info("Training vision model...")
            
            # Load vision data
            from app_data_utils import load_vision_data
            vision_data = load_vision_data(self.args.data)
            
            # Train vision model
            vision_output = os.path.join(self.args.output, 'vision')
            vision_results = model_manager.train_model(
                'vision', 
                vision_data,  # Pass the entire dictionary
                None,  # No separate val_data needed since it's in the dictionary
                vision_output
            )

            self.logger.info(f"Vision model training complete! Model saved to: {vision_results['model_path']}")
        
        # Train sensor model if requested
        if self.args.sensor or not (self.args.vision or self.args.sensor):
            self.logger.info("Training sensor model...")
            
            # Load sensor data
            from app_data_utils import load_sensor_data
            sensor_data = load_sensor_data(self.args.data)
            
            # Train sensor model
            sensor_output = os.path.join(self.args.output, 'sensor')
            sensor_results = model_manager.train_model(
                'sensor', 
                (sensor_data['X_train'], sensor_data['y_train']),
                (sensor_data['X_val'], sensor_data['y_val']),
                sensor_output
            )
            
            self.logger.info(f"Sensor model training complete! Model saved to: {sensor_results['model_path']}")
        
        # Export models for Unity if requested
        if self.args.export_unity:
            self.logger.info("Exporting models for Unity...")
            
            unity_export_dir = os.path.join(self.args.output, 'unity_models')
            os.makedirs(unity_export_dir, exist_ok=True)
            
            if self.args.vision or not (self.args.vision or self.args.sensor):
                from app_model_export import export_model_to_onnx
                vision_model_path = self.config_manager.get_value('model_paths.vision')
                
                if vision_model_path and os.path.exists(vision_model_path):
                    vision_model = tf.keras.models.load_model(vision_model_path)
                    export_model_to_onnx(
                        vision_model,
                        os.path.join(unity_export_dir, 'vision_model.onnx')
                    )
            
            if self.args.sensor or not (self.args.vision or self.args.sensor):
                from app_model_export import export_model_to_onnx
                sensor_model_path = self.config_manager.get_value('model_paths.sensor')
                
                if sensor_model_path and os.path.exists(sensor_model_path):
                    sensor_model = tf.keras.models.load_model(sensor_model_path)
                    export_model_to_onnx(
                        sensor_model,
                        os.path.join(unity_export_dir, 'sensor_model.onnx')
                    )
            
            self.logger.info(f"Model export complete! Models saved to: {unity_export_dir}")
    
    def _cmd_test(self):
        """Execute test command."""
        self.logger.info("Testing models...")
        
        # Import testing module
        from tool_test_models import main as test_main
        
        # Set up command-line arguments for test_main
        sys.argv = [
            sys.argv[0],
            '--config', self.args.config
        ]
        
        if self.args.vision_model:
            sys.argv.extend(['--vision-model', self.args.vision_model])
        
        if self.args.sensor_model:
            sys.argv.extend(['--sensor-model', self.args.sensor_model])
        
        if self.args.image:
            sys.argv.extend(['--image', self.args.image])
        
        if self.args.sensor_data:
            sys.argv.extend(['--sensor-data', self.args.sensor_data])
        
        if self.args.scaler:
            sys.argv.extend(['--scaler', self.args.scaler])
        
        if self.args.output:
            sys.argv.extend(['--output', self.args.output])
        
        # Run test_main
        test_main()
        
        self.logger.info(f"Testing complete! Results saved to: {self.args.output}")
    
    def _cmd_run(self):
        """Execute run command."""
        self.logger.info("Running monitoring application...")
        
        # Import application module
        from app import app, socketio
        
        # Run application
        socketio.run(
            app, 
            host=self.args.host, 
            port=self.args.port, 
            debug=self.args.debug,
            allow_unsafe_werkzeug=True
        )

if __name__ == '__main__':
    cli = AssemblyML()
    cli.run()