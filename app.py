import time
import threading
import queue
import pandas as pd
import numpy as np
import cv2
import json
import io
import os
import logging
import base64
from PIL import Image
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from app_base_processor import DataProcessingError, PredictionError
from app_vision_processor import VisionProcessor
from app_sensor_processor import SensorProcessor
from app_config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application state class
class ApplicationState:
    """Thread-safe application state manager."""
    
    def __init__(self):
        """Initialize application state."""
        self._lock = threading.RLock()
        self._state = {
            'vision_analysis': None,
            'sensor_analysis': None,
            'combined_analysis': None,
            'last_update_time': None,
            'system_status': 'stopped',
            'sensor_readings': []
        }
        self._max_sensor_readings = 500
    
    def update(self, key, value):
        """
        Update state value.
        
        Args:
            key (str): State key
            value: New value
        """
        with self._lock:
            self._state[key] = value
            self._state['last_update_time'] = time.time()
    
    def get(self, key, default=None):
        """
        Get state value.
        
        Args:
            key (str): State key
            default: Default value if key doesn't exist
            
        Returns:
            Value for key or default
        """
        with self._lock:
            return self._state.get(key, default)
    
    def get_all(self):
        """
        Get complete state.
        
        Returns:
            dict: Complete application state
        """
        with self._lock:
            return self._state.copy()
    
    def add_sensor_reading(self, reading):
        """
        Add sensor reading to buffer.
        
        Args:
            reading (dict): Sensor reading
        """
        with self._lock:
            self._state['sensor_readings'].append(reading)
            
            # Enforce maximum buffer size
            if len(self._state['sensor_readings']) > self._max_sensor_readings:
                self._state['sensor_readings'] = self._state['sensor_readings'][-self._max_sensor_readings:]
    
    def get_sensor_readings(self, count=None):
        """
        Get sensor readings.
        
        Args:
            count (int, optional): Number of readings to return
            
        Returns:
            list: Sensor readings
        """
        with self._lock:
            readings = self._state['sensor_readings']
            if count:
                return readings[-count:]
            return readings.copy()
    
    def reset(self):
        """Reset application state."""
        with self._lock:
            self._state['vision_analysis'] = None
            self._state['sensor_analysis'] = None
            self._state['combined_analysis'] = None
            self._state['last_update_time'] = time.time()

# History database class using SQLite
class HistoryDatabase:
    """Database for storing process history."""
    
    def __init__(self, db_path='history.db'):
        """
        Initialize history database.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        import sqlite3
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    timestamp INTEGER NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()
    
    def add_event(self, event_type, details=None, metadata=None):
        """
        Add event to history.
        
        Args:
            event_type (str): Type of event
            details (str): Event details
            metadata (dict): Additional metadata
        """
        import sqlite3
        timestamp = int(time.time() * 1000)  # Milliseconds
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO events (event_type, details, timestamp, metadata) VALUES (?, ?, ?, ?)',
                    (event_type, details, timestamp, json.dumps(metadata) if metadata else None)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
    
    def get_events(self, limit=100, event_type=None, start_time=None, end_time=None):
        """
        Get events from history.
        
        Args:
            limit (int): Maximum number of events to return
            event_type (str): Filter by event type
            start_time (int): Filter by start time (milliseconds)
            end_time (int): Filter by end time (milliseconds)
            
        Returns:
            list: List of event dictionaries
        """
        import sqlite3
        query = 'SELECT id, event_type, details, timestamp, metadata FROM events'
        params = []
        
        # Build WHERE clause
        where_clauses = []
        if event_type:
            where_clauses.append('event_type = ?')
            params.append(event_type)
        
        if start_time:
            where_clauses.append('timestamp >= ?')
            params.append(start_time)
        
        if end_time:
            where_clauses.append('timestamp <= ?')
            params.append(end_time)
        
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        # Add ORDER BY and LIMIT
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        try:
            # Execute query
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                events = []
                for row in cursor:
                    event = {
                        'id': row['id'],
                        'event': row['event_type'],
                        'details': row['details'],
                        'timestamp': row['timestamp']
                    }
                    
                    # Parse metadata if available
                    if row['metadata']:
                        try:
                            metadata = json.loads(row['metadata'])
                            event.update(metadata)
                        except json.JSONDecodeError:
                            pass
                    
                    events.append(event)
                
                return events
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return []

    def check_connection(self):
        """
        Check if the database connection is working.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            import sqlite3
            with sqlite3.connect(self.db_path, timeout=1.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                return True
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return False

# Monitoring thread class
class MonitoringThread(threading.Thread):
    """Thread for monitoring the task."""
    
    def __init__(self, name, interval, callback, app_state, logger):
        """
        Initialize monitoring thread.
        
        Args:
            name (str): Thread name
            interval (float): Monitoring interval in seconds
            callback (callable): Function to call on each iteration
            app_state (ApplicationState): Application state
            logger (logging.Logger): Logger instance
        """
        super().__init__(name=name, daemon=True)
        self.interval = interval
        self.callback = callback
        self.app_state = app_state
        self.logger = logger
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
    
    def run(self):
        """Run the monitoring thread."""
        self.logger.info(f"Starting {self.name} thread")
        while not self.stop_event.is_set():
            try:
                # Call the monitoring callback
                result = self.callback()
                
                # Put result in queue if not None
                if result is not None:
                    self.data_queue.put(result)
                
                # Sleep for interval
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error in {self.name} thread: {e}")
                # Brief sleep to avoid tight error loops
                time.sleep(0.1)
        
        self.logger.info(f"{self.name} thread stopped")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.stop_event.set()
    
    def get_data(self, block=False, timeout=None):
        """
        Get data from the thread's queue.
        
        Args:
            block (bool): Whether to block until data is available
            timeout (float): Timeout for blocking get
            
        Returns:
            Data from queue or None if queue is empty
        """
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

class TaskMonitor:
    def __init__(self, config_dict):
        """
        Initializes the task monitoring system that validates task steps
        by combining computer vision and sensor data analysis.
        
        Args:
            config_dict (dict): System configuration
        """
        self.logger = logging.getLogger('task_monitor')
        self.config_manager = ConfigManager()
        
        # Initialize attributes to None to ensure they exist
        self.app_state = None
        self.history_db = None
        self.vision_system = None
        self.sensor_processor = None
        self.camera_captures = []
        self.is_running = False
        self.monitoring_threads = []
        
        try:
            # Load the provided configuration
            self._load_configuration(config_dict)
            
            # Initialize application state
            self.app_state = ApplicationState()
            
            # Initialize history database
            self.history_db = HistoryDatabase(db_path='data/history.db')
            
            # Initialize the computer vision system for step detection
            try:
                num_steps = len(self.config_manager.get_value('step_definitions', []))
                vision_model_path = self.config_manager.get_value('model_paths.vision')
                
                self.logger.info(f"Initializing vision system with {num_steps} steps")
                self.logger.info(f"Vision model path: {vision_model_path}")
                
                if not vision_model_path or not os.path.exists(vision_model_path):
                    self.logger.error(f"Vision model not found at path: {vision_model_path}")
                    raise ValueError(f"Vision model not found at: {vision_model_path}")
                    
                self.vision_system = VisionProcessor(
                    num_process_steps=num_steps,
                    model_path=vision_model_path,
                    logger=logging.getLogger('vision_processor')
                )
                
                # Verify model was loaded successfully
                if self.vision_system.model is None:
                    raise ValueError("Failed to load vision model")
                    
                self.logger.info(f"Vision system initialized successfully with model: {vision_model_path}")
            except Exception as e:
                self.logger.error(f"Error initializing vision system: {e}")
                self.vision_system = None
            
            # Initialize the sensor data analysis system
            try:
                # Use ConfigManager's get_value method
                sensor_config = self.config_manager.get_value('sensor_config', {})
                sensor_model_path = self.config_manager.get_value('model_paths.sensor')
                
                self.logger.info(f"Initializing sensor system with {len(sensor_config)} sensors")
                self.sensor_processor = SensorProcessor(
                    sensor_config=sensor_config,
                    model_path=sensor_model_path,
                    logger=logging.getLogger('sensor_processor')
                )
                
                # Load scalers if provided
                scaler_path = self.config_manager.get_value('model_paths.sensor_scalers')
                if scaler_path and os.path.exists(scaler_path):
                    self.sensor_processor.load_scalers(scaler_path)
            except Exception as e:
                self.logger.error(f"Error initializing sensor system: {e}")
                self.sensor_processor = None
            
            # Initialize video capture for each camera
            camera_sources = self.config_manager.get_value('camera_sources', [])
            for camera_source in camera_sources:
                try:
                    cap = cv2.VideoCapture(camera_source)
                    if not cap.isOpened():
                        self.logger.warning(f"Could not initialize camera at source: {camera_source}")
                    self.camera_captures.append(cap)
                except Exception as e:
                    self.logger.error(f"Error initializing camera {camera_source}: {e}")
            
            # Ensure application state is updated
            if self.app_state:
                self.app_state.update('system_status', 'initialized')

        except Exception as e:
            # Ensure logging even if other initializations fail
            self.logger.error(f"Critical error during TaskMonitor initialization: {e}")
            raise

    def _load_configuration(self, config_dict):
        """
        Load and validate the configuration.
        
        Args:
            config_dict (dict): Configuration dictionary to load
        """
        try:
            # Merge the provided config with existing configuration
            def deep_merge(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        base[key] = deep_merge(base[key], value)
                    elif isinstance(value, list) and key in base and isinstance(base[key], list):
                        # For lists, replace entirely
                        base[key] = value
                    else:
                        base[key] = value
                return base
            
            # Merge the provided config with defaults from ConfigManager
            merged_config = deep_merge(self.config_manager.config.copy(), config_dict)
            
            # Update ConfigManager's configuration
            for key, value in merged_config.items():
                self.config_manager.config[key] = value
            
            # Log configuration details
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise ValueError(f"Invalid configuration: {e}")
   
    def start_monitoring(self):
        """Starts all monitoring subsystems in separate threads."""
        if self.is_running:
            self.logger.warning("Monitoring system is already running")
            return False
        
        self.is_running = True
        
        # Start computer vision monitoring thread if vision system is available
        if self.vision_system:
            vision_thread = MonitoringThread(
                name="vision_monitoring",
                interval=0.1,  # 10 FPS
                callback=self._run_vision_monitoring,
                app_state=self.app_state,
                logger=self.logger
            )
            vision_thread.start()
            self.monitoring_threads.append(vision_thread)
        
        # Start sensor data monitoring thread if sensor processor is available
        if self.sensor_processor:
            sensor_thread = MonitoringThread(
                name="sensor_monitoring",
                interval=0.01,  # 100 Hz
                callback=self._run_sensor_monitoring,
                app_state=self.app_state,
                logger=self.logger
            )
            sensor_thread.start()
            self.monitoring_threads.append(sensor_thread)
        
        # Start data integration thread
        integration_thread = MonitoringThread(
            name="data_integration",
            interval=0.5,  # 2 Hz
            callback=self._run_data_integration,
            app_state=self.app_state,
            logger=self.logger
        )
        integration_thread.start()
        self.monitoring_threads.append(integration_thread)
        
        # Update application state
        self.app_state.update('system_status', 'running')
        
        # Add to history
        self.history_db.add_event(
            event_type='start',
            details='Task monitoring system started',
            metadata={
                'camera_count': len(self.camera_captures),
                'sensor_count': len(self.config_manager.get_value('sensor_config', {})),
                'step_count': len(self.config_manager.get_value('step_definitions', []))
            }
        )
                
        self.logger.info("Task monitoring system started successfully")
        return True
    
    def stop_monitoring(self):
        """Safely stops all monitoring subsystems and releases resources."""
        if not self.is_running:
            self.logger.warning("Monitoring system is not running")
            return False
        
        self.is_running = False
        
        # Stop all monitoring threads
        for thread in self.monitoring_threads:
            thread.stop()
            thread.join(timeout=1.0)
        
        # Clear the thread list
        self.monitoring_threads = []
        
        # Release camera resources
        for camera in self.camera_captures:
            camera.release()
        
        # Update application state
        self.app_state.update('system_status', 'stopped')
        
        # Add to history
        self.history_db.add_event(
            event_type='stop',
            details='Task monitoring system stopped'
        )
        
        self.logger.info("Task monitoring system stopped")
        return True
    
    def _run_vision_monitoring(self):
        """
        Continuously monitors task steps using computer vision.
        Runs at approximately 10 frames per second.
        
        Returns:
            dict: Vision analysis results or None if no results
        """
        if not self.vision_system or not self.camera_captures:
            return None
        
        vision_results = {}
        
        # Process frames from each camera
        for camera_idx, camera in enumerate(self.camera_captures):
            ret, frame = camera.read()
            if not ret:
                continue
            
            try:
                # Detect process steps in the current frame
                step_detection = self.vision_system.detect_steps(frame)
                vision_results[f'camera_{camera_idx}'] = step_detection
                
                # Convert numpy types to Python native types for JSON serialization
                for key, value in step_detection.items():
                    if isinstance(value, np.ndarray):
                        step_detection[key] = value.tolist()
                
            except Exception as e:
                self.logger.error(f"Error in vision processing for camera {camera_idx}: {e}")
        
        # Update system state if we have new results
        if vision_results:
            self.app_state.update('vision_analysis', vision_results)
            self.app_state.update('last_update_time', time.time())
            
            # Return results for event emission
            return {
                'type': 'vision_update',
                'data': {
                    'results': vision_results,
                    'timestamp': time.time() * 1000
                }
            }
        
        return None
    
    def _run_sensor_monitoring(self):
        """
        Continuously collects and analyzes sensor data to validate process steps.
        Runs at approximately 100Hz sampling rate.
        
        Returns:
            dict: Sensor analysis results or None if no results
        """
        if not self.sensor_processor:
            return None
        
        # In production, this would read from physical sensors
        # Here, we generate simulated data for demonstration
        simulated_reading = self._generate_simulated_sensor_reading()
        
        # Maintain a rolling buffer of sensor readings
        self.app_state.add_sensor_reading(simulated_reading)
        
        # Process sensor data when sufficient samples are available
        sensor_readings = self.app_state.get_sensor_readings()
        if len(sensor_readings) >= 100:
            try:
                # Convert to DataFrame
                sensor_dataframe = pd.DataFrame(sensor_readings)
                
                # Process sensor data
                step_detection = self.sensor_processor.detect_steps(sensor_dataframe)
                
                # Convert numpy types to Python native types for JSON serialization
                for key, value in step_detection.items():
                    if isinstance(value, np.ndarray):
                        step_detection[key] = value.tolist()
                
                # Update system state
                self.app_state.update('sensor_analysis', step_detection)
                
                # Return results for event emission
                return {
                    'type': 'sensor_update',
                    'data': {
                        'detection': step_detection,
                        'latest_readings': sensor_readings[-10:],  # Send last 10 readings
                        'timestamp': time.time() * 1000
                    }
                }
            except Exception as e:
                self.logger.error(f"Error in sensor processing: {e}")
        
        return None
    
    def _run_data_integration(self):
        """
        Combines vision and sensor data to determine completed process steps.
        Runs at approximately 2Hz update rate.
        
        Returns:
            dict: Combined analysis results or None if no results
        """
        vision_results = self.app_state.get('vision_analysis')
        sensor_results = self.app_state.get('sensor_analysis')
        step_definitions = self.config_manager.get_value('step_definitions', [])

        if vision_results and sensor_results:
            try:
                # Fuse vision and sensor data
                combined_analysis = self._combine_vision_and_sensor_data(vision_results, sensor_results)
                
                # Update system state
                self.app_state.update('combined_analysis', combined_analysis)
                
                # Format for event emission
                formatted_result = {
                    'completedSteps': combined_analysis['completed_steps'],
                    'stepDefinitions': step_definitions,
                    'stepConfidence': {
                        step_definitions.get(step_id, f"Step {step_id}"): {'combined': score}
                        for step_id, score in combined_analysis['confidence'].items()
                    },
                    'allCompleted': combined_analysis['all_completed'],
                    'timestamp': time.time() * 1000
                }
                
                return {
                    'type': 'state_update',
                    'data': formatted_result
                }
            except Exception as e:
                self.logger.error(f"Error in data integration: {e}")
        
        return None
    
    def _combine_vision_and_sensor_data(self, vision_results, sensor_results):
        """
        Integrates vision and sensor analysis to validate process step completion.
        
        Args:
            vision_results: Dictionary of vision analysis from all cameras
            sensor_results: Dictionary of sensor analysis results
            
        Returns:
            Dictionary containing completed steps, confidence scores, etc.
        """
        # Get step definitions
        step_definitions = self.config_manager.get_value('step_definitions', [])
        all_process_steps = list(range(len(step_definitions)))
        
        # Get steps marked complete by vision system (union across all cameras)
        vision_completed_steps = set()
        for camera_result in vision_results.values():
            if isinstance(camera_result, dict) and 'completed_steps' in camera_result:
                vision_completed_steps.update(camera_result['completed_steps'])
        
        # Get steps marked complete by sensor system
        sensor_completed_steps = set()
        if isinstance(sensor_results, dict) and 'completed_steps' in sensor_results:
            sensor_completed_steps.update(sensor_results['completed_steps'])
        
        # Steps are only confirmed when both systems agree
        confirmed_completed_steps = list(vision_completed_steps.intersection(sensor_completed_steps))
        
        # Calculate weighted confidence for each step
        step_confidence_scores = {}
        for step in all_process_steps:
            # Average vision confidence across all camera views
            vision_confidence = 0
            camera_count = 0
            for camera_result in vision_results.values():
                if isinstance(camera_result, dict) and 'predictions' in camera_result:
                    predictions = camera_result['predictions']
                    if isinstance(predictions, list) and step < len(predictions):
                        vision_confidence += predictions[step]
                        camera_count += 1
            
            vision_confidence = vision_confidence / camera_count if camera_count > 0 else 0
            
            # Get sensor confidence for this step
            sensor_confidence = 0
            if (isinstance(sensor_results, dict) and 'predictions' in sensor_results and
                    isinstance(sensor_results['predictions'], list) and 
                    step < len(sensor_results['predictions'])):
                sensor_confidence = sensor_results['predictions'][step]
            
            # Weighted average (60% vision, 40% sensor)
            step_confidence_scores[step] = 0.6 * vision_confidence + 0.4 * sensor_confidence
        
        return {
            'completed_steps': confirmed_completed_steps,
            'all_completed': set(confirmed_completed_steps) == set(all_process_steps),
            'confidence': step_confidence_scores,
            'step_details': {
                step_id: {
                    'description': step_definitions[step_id] if step_id < len(step_definitions) else f"Step {step_id}",
                    'is_complete': step_id in confirmed_completed_steps,
                    'confidence_score': step_confidence_scores.get(step_id, 0)
                }
                for step_id in all_process_steps
            }
        }
    
    def analyze_frame(self, frame):
        """
        Analyzes a single frame for process step detection.
        This is used for the on-demand analysis API endpoint.
        
        Args:
            frame (np.array): Input BGR image from camera or uploaded image
            
        Returns:
            dict: Contains detection results and processing metadata
        """
        if not self.vision_system:
            raise RuntimeError("Vision system not initialized")
        
        # Get step definitions
        step_definitions = self.config_manager.get_value('step_definitions', [])
        
        # Get vision analysis for this frame
        try:
            vision_results = self.vision_system.detect_steps(frame)
            
            self.logger.info(f"Vision predictions: {vision_results['predictions']}")
            self.logger.info(f"Completed steps: {vision_results['completed_steps']}")

            # Convert numpy types to Python native types for JSON serialization
            for key, value in vision_results.items():
                if isinstance(value, np.ndarray):
                    vision_results[key] = value.tolist()
            
            # Get the most recent sensor analysis if available
            sensor_results = self.app_state.get('sensor_analysis')
            
            if sensor_results:
                # If we have sensor data, perform combined analysis
                combined_results = self._combine_vision_and_sensor_data(
                    {'single_camera': vision_results}, 
                    sensor_results
                )
                return combined_results
            else:
                # If no sensor data, return vision-only results
                return {
                    'completed_steps': vision_results['completed_steps'],
                    'all_completed': vision_results['all_completed'],
                    'confidence': {
                        step_id: vision_results['predictions'][step_id] 
                        for step_id in range(len(vision_results['predictions']))
                    },
                    'step_details': {
                        step_id: {
                            'description': step_definitions[step_id] if step_id < len(step_definitions) else f"Step {step_id}",
                            'is_complete': step_id in vision_results['completed_steps'],
                            'confidence_score': vision_results['predictions'][step_id]
                        }
                        for step_id in range(len(vision_results['predictions']))
                    }
                }
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {e}")
            raise
    
    def _generate_simulated_sensor_reading(self):
        """
        Generates realistic sensor readings for demonstration purposes.
        In a production system, this would interface with actual sensors.
        
        Returns:
            Dictionary containing simulated sensor values with timestamp
        """
        # Get sensor names from config
        sensor_config = self.config_manager.get_value('sensor_config', {})
        sensor_reading = {'timestamp': time.time()}
        
        # Generate realistic values for each sensor
        for sensor_name, sensor_info in sensor_config.items():
            # Use expected range if available, otherwise default values
            expected_range = sensor_info.get('expected_range', [0, 100])
            min_val, max_val = expected_range
            mean_val = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / 10
            
            # Generate normal distribution value within expected range
            value = np.random.normal(mean_val, std_dev)
            sensor_reading[sensor_name] = float(value)
        
        return sensor_reading

    def get_current_status(self):
        """
        Retrieves the current state of the task monitoring.
        
        Returns:
            Dictionary containing the latest analysis along with timestamp.
        """
        # Get combined analysis
        combined = self.app_state.get('combined_analysis', {})
        
        # Fix: Ensure combined is not None
        if combined is None:
            combined = {}  # Use empty dict if combined is None
        
        # Get process step definitions using ConfigManager
        step_definitions = self.config_manager.get_value('step_definitions', [])
        
        # Format for client consumption
        return {
            'completedSteps': combined.get('completed_steps', []),
            'stepDefinitions': step_definitions,
            'stepConfidence': {
                step_definitions[step_id] if step_id < len(step_definitions) else f"Step {step_id}": 
                {'combined': score}
                for step_id, score in combined.get('confidence', {}).items()
            } if combined else {},
            'allCompleted': combined.get('all_completed', False),
            'timestamp': int(self.app_state.get('last_update_time', time.time()) * 1000),
            'systemStatus': self.app_state.get('system_status', 'unknown')
        }

    def reset_process(self):
        """
        Resets the current process without stopping the monitoring system.
        
        Returns:
            bool: True if reset was successful
        """
        # Reset application state
        self.app_state.reset()
        
        # Add reset event to history
        self.history_db.add_event(
            event_type='reset',
            details='Task reset'
        )
        
        self.logger.info("Task reset")
        return True

# Flask web service for remote monitoring and control
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instance of TaskMonitor
task_monitor = None

# Helper functions
def _emit_event(event_type, data):
    """
    Emit event to all connected clients.
    
    Args:
        event_type (str): Event type
        data (dict): Event data
    """
    socketio.emit(event_type, {
        'type': event_type,
        'data': data
    })

def _add_history_event(event_type, details, metadata=None):
    """
    Add event to history and broadcast to clients.
    
    Args:
        event_type (str): Event type
        details (str): Event details
        metadata (dict, optional): Additional metadata
    """
    if task_monitor and task_monitor.history_db:
        # Add to database
        task_monitor.history_db.add_event(event_type, details, metadata)
        
        # Create event object for broadcasting
        event = {
            'event': event_type,
            'details': details,
            'timestamp': int(time.time() * 1000)
        }
        if metadata:
            event.update(metadata)
        
        # Broadcast to all connected clients
        _emit_event('history_update', event)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect(data=None):
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial data to newly connected client
    if task_monitor:
        try:
            emit('initial_data', {
                'type': 'initialData',
                'data': {
                    'currentState': task_monitor.get_current_status(),
                    'history': task_monitor.history_db.get_events(limit=10),
                    'alerts': []  # No alerts on initial connection
                }
            })
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
            emit('server_message', {
                'type': 'error',
                'message': 'Error preparing initial data'
            })
    else:
        emit('server_message', {
            'type': 'error',
            'message': 'Monitoring system not initialized'
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_history')
def handle_request_history(data=None):
    """
    Handle request for history data.
    
    Args:
        data (dict, optional): Request parameters (limit, event_type, etc.)
    """
    if not task_monitor:
        emit('error', {'message': 'Monitoring system not initialized'})
        return
    
    # Parse parameters
    limit = data.get('limit', 100) if data else 100
    event_type = data.get('event_type') if data else None
    
    # Get history events
    events = task_monitor.history_db.get_events(
        limit=limit,
        event_type=event_type
    )
    
    # Send response
    emit('history_data', {
        'type': 'historyData',
        'data': events
    })

@socketio.on('reset_process')
def handle_reset_process():
    """Handle request to reset the process."""
    if not task_monitor:
        emit('error', {'message': 'Monitoring system not initialized'})
        return
    
    # Reset process
    if task_monitor.reset_process():
        # Notify all clients
        _emit_event('alert', {
            'type': 'success',
            'message': 'Task has been reset successfully',
            'timestamp': time.time() * 1000
        })
        
        emit('reset_success', {'status': 'success'})
    else:
        emit('reset_failure', {'status': 'error', 'message': 'Failed to reset process'})

@socketio.on('send_alert')
def handle_send_alert(data):
    """
    Allows clients to send alerts that will be broadcast to all clients.
    
    Args:
        data (dict): Alert data (type, message)
    """
    if not isinstance(data, dict) or 'message' not in data:
        emit('error', {'message': 'Invalid alert format'})
        return
    
    alert_type = data.get('type', 'info')
    alert = {
        'type': alert_type,
        'message': data['message'],
        'timestamp': time.time() * 1000
    }
    
    # Add to history if it's a significant alert
    if alert_type in ['error', 'warning', 'success']:
        _add_history_event(
            event_type='alert',
            details=data['message'],
            metadata={'alertType': alert_type}
        )
    
    # Broadcast to all clients
    _emit_event('alert', alert)
    
    emit('alert_sent', {'status': 'success'})

# Flask routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/status', methods=['GET'])
def get_process_status():
    """Returns the current status of the task monitoring."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 500
    
    try:
        return jsonify(task_monitor.get_current_status())
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start', methods=['POST'])
def start_task_monitoring():
    """Initializes and starts the task monitoring system."""
    global task_monitor
    
    try:
        # Get configuration from request
        config = request.json
        if not config:
            return jsonify({'error': 'Configuration required'}), 400
        
        # Create and start process monitor
        task_monitor = TaskMonitor(config)
        if task_monitor and task_monitor.vision_system:
            logger.info(f"Vision system initialized with {len(task_monitor.config_manager.get_value('step_definitions', []))} steps")
            logger.info(f"Step definitions: {task_monitor.config_manager.get_value('step_definitions', [])}")
            logger.info(f"Model path: {task_monitor.config_manager.get_value('model_paths.vision')}")
        else:
            logger.error("Vision system was not properly initialized")

        if not task_monitor.start_monitoring():
            return jsonify({'error': 'Failed to start monitoring'}), 500
        
        # Notify all clients
        _emit_event('system_status', {
            'status': 'started',
            'message': 'Process monitoring started successfully',
            'timestamp': time.time() * 1000
        })
        
        return jsonify({'message': 'Process monitoring started successfully'})
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_task_monitoring():
    """Stops the task monitoring system and releases resources."""
    global task_monitor
    
    try:
        if task_monitor:
            if not task_monitor.stop_monitoring():
                return jsonify({'error': 'Failed to stop monitoring'}), 500
            
            task_monitor = None
            
            # Notify all clients
            _emit_event('system_status', {
                'status': 'stopped',
                'message': 'Process monitoring stopped successfully',
                'timestamp': time.time() * 1000
            })
        
        return jsonify({'message': 'Process monitoring stopped successfully'})
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_monitoring():
    """Resets the current process without stopping the monitoring system."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 400
    
    try:
        # Reset process
        if not task_monitor.reset_process():
            return jsonify({'error': 'Failed to reset process'}), 500
        
        # Notify all clients
        _emit_event('system_status', {
            'status': 'reset',
            'message': 'Task has been reset successfully',
            'timestamp': time.time() * 1000
        })
        
        return jsonify({'message': 'Task reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Accepts an image frame and returns process step detection results.
    
    Request format:
    - JSON with 'image' field containing base64 encoded image
    - OR multipart form with 'image' file upload
    
    Returns:
    - JSON with detection results and processing metadata
    """
    if not task_monitor:
        return jsonify({'status': 'error', 'message': 'Monitoring system not initialized'}), 400
    
    try:
        # Debug the request content type
        logger.info(f"analyze-frame request content type: {request.content_type}")
        
        # Ensure static directory exists for saving debug image
        static_dir = 'static'
        os.makedirs(static_dir, exist_ok=True)
        debug_image_path = os.path.join(static_dir, 'uploaded.png')
        
        # Handle both JSON (base64) and multipart form uploads
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if not data or 'image' not in data:
                logger.warning("Missing image data in JSON request")
                return jsonify({'status': 'error', 'message': 'Missing image data'}), 400
            
            # Decode base64 image
            try:
                # Handle both with and without the data:image prefix
                image_data = data['image']
                if ',' in image_data:
                    image_data = image_data.split(',', 1)[1]
                
                image_data = base64.b64decode(image_data)
                logger.info(f"Base64 image decoded, size: {len(image_data)} bytes")
                
                # Read with OpenCV directly to avoid PIL -> OpenCV conversion issues
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # This reads directly in BGR format
                
                if frame is None:
                    logger.error("Failed to decode image data into OpenCV format")
                    return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400
                
                # Save the debug image
                cv2.imwrite(debug_image_path, frame)
                logger.info(f"Saved debug image to {debug_image_path}")
                
                logger.info(f"Decoded image with shape: {frame.shape}")
            except Exception as e:
                logger.error(f"Invalid base64 image data: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Invalid image data: {str(e)}'}), 400
        else:
            # Check if 'image' is in files
            if 'image' not in request.files:
                logger.warning("No image file in multipart request")
                return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
            
            # Read uploaded file
            try:
                image_file = request.files['image']
                logger.info(f"Received image file: {image_file.filename}, size: {image_file.content_length or 'unknown'} bytes")
                
                # Save the file directly to the static directory for debugging
                image_file.save(debug_image_path)
                logger.info(f"Saved debug image to {debug_image_path}")
                
                # Load directly with OpenCV - this reads in BGR format as expected by OpenCV functions
                frame = cv2.imread(debug_image_path)
                
                if frame is None:
                    logger.error(f"Failed to load image from path: {debug_image_path}")
                    return jsonify({'status': 'error', 'message': 'Invalid image file format'}), 400
                
                logger.info(f"Loaded image with shape: {frame.shape}")
            except Exception as e:
                logger.error(f"Invalid image file: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Invalid image file: {str(e)}'}), 400
        
        # Debug the frame shape
        logger.info(f"Processing OpenCV frame with shape: {frame.shape}")
        
        # Analyze the frame
        try:
            results = task_monitor.analyze_frame(frame)
            logger.info(f"Analysis complete: {len(results.get('completed_steps', []))} steps detected")
        except Exception as e:
            logger.error(f"Error in analyze_frame method: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f"Error analyzing image: {str(e)}",
                'timestamp': time.time()
            }), 500
        
        # Add analysis event to history
        _add_history_event(
            event_type='Image Analysis',
            details=f"{len(results.get('completed_steps', []))} steps detected",
            metadata={
                'confidence': (1.0 if results.get('all_completed', False) else 
                              sum(results.get('confidence', {}).values()) / len(results.get('confidence', {}))
                              if results.get('confidence', {}) else 0)
            }
        )
        
        # Emit analysis results to all clients
        _emit_event('frame_analysis', {
            'results': results,
            'timestamp': time.time() * 1000
        })
        
        # Include the debug image URL in the response
        return jsonify({
            'status': 'success',
            'results': results,
            'debug_image_url': f'/static/uploaded.png?t={int(time.time())}',
            'timestamp': time.time()
        })
    
    except DataProcessingError as e:
        error_message = str(e)
        logger.warning(f"Data processing error: {error_message}")
        
        # Emit error to all clients
        _emit_event('alert', {
            'type': 'warning',
            'message': f"Error processing image: {error_message}",
            'timestamp': time.time() * 1000
        })
        
        return jsonify({
            'status': 'error',
            'message': error_message,
            'timestamp': time.time()
        }), 400
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error analyzing frame: {error_message}", exc_info=True)
        
        # Emit error to all clients
        _emit_event('alert', {
            'type': 'error',
            'message': f"Error analyzing image: {error_message}",
            'timestamp': time.time() * 1000
        })
        
        return jsonify({
            'status': 'error',
            'message': error_message,
            'timestamp': time.time()
        }), 500

# API routes for cleaner REST interface
@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint for getting current system status."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 500
    
    try:
        return jsonify(task_monitor.get_current_status())
    except Exception as e:
        logger.error(f"Error in API status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint for getting process history."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 500
    
    try:
        # Parse query parameters
        limit = request.args.get('limit', default=100, type=int)
        event_type = request.args.get('event_type', default=None, type=str)
        
        # Get history events
        events = task_monitor.history_db.get_events(
            limit=limit,
            event_type=event_type
        )
        
        return jsonify(events)
    except Exception as e:
        logger.error(f"Error in API history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """API endpoint for getting system alerts."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 500
    
    try:
        # Get alert events
        alerts = task_monitor.history_db.get_events(
            limit=20,
            event_type='alert'
        )
        
        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Error in API alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for checking system health."""
    global task_monitor, app_start_time
    
    # If app_start_time is not defined, define it now
    if 'app_start_time' not in globals():
        app_start_time = time.time()
    
    if not task_monitor:
        return jsonify({
            'status': 'error',
            'message': 'Monitoring system not initialized',
            'uptime': time.time() - app_start_time
        }), 503  # Service Unavailable
    
    try:
        # Comprehensive system health check
        status = {
            'status': 'ok',
            'system': task_monitor.is_running,
            'vision_model': task_monitor.vision_system is not None,
            'sensor_model': task_monitor.sensor_processor is not None,
            'db': task_monitor.history_db.check_connection() if task_monitor.history_db else False,
            'uptime': time.time() - app_start_time
        }
        
        # If any critical component is down, return degraded status
        if not status['vision_model'] or not status['sensor_model'] or not status['db']:
            status['status'] = 'degraded'
            return jsonify(status), 207  # Multi-Status
            
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'uptime': time.time() - app_start_time
        }), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """API endpoint for resetting the process."""
    if not task_monitor:
        return jsonify({'error': 'Monitoring system not initialized'}), 400
    
    try:
        # Reset process
        if not task_monitor.reset_process():
            return jsonify({'error': 'Failed to reset process'}), 500
        
        return jsonify({'status': 'success', 'message': 'Task reset successfully'})
    except Exception as e:
        logger.error(f"Error in API reset: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Main entry point
if __name__ == '__main__':
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process Monitoring App')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--config', help='Path to configuration file to auto-start monitoring')
    args = parser.parse_args()
    
    # Load configuration and auto-start if config file is provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Create and start process monitor
            task_monitor = TaskMonitor(config)
            task_monitor.start_monitoring()
            logger.info(f"Monitoring auto-started with config: {args.config}")
        except Exception as e:
            logger.error(f"Error auto-starting: {e}")
    
    # Run the Flask app with Socket.IO
    socketio.run(
        app, 
        host=args.host, 
        port=args.port, 
        debug=args.debug, 
        allow_unsafe_werkzeug=True
    )