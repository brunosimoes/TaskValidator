# Task Validator

A real-time monitoring application that utilizes computer vision and sensor data analysis to track task completion.


## AI Models

### Vision Model Architecture

The vision system uses transfer learning with MobileNetV2 as its base model. This was chosen for its efficiency and ability to run on edge devices  while maintaining high accuracy.

Here's the model architecture:

1. **Base Model**: MobileNetV2 pre-trained on ImageNet

2. **Custom Classification Head**:
   - Global Average Pooling to reduce feature dimensions
   - Dense layer with 256 neurons and ReLU activation
   - Dropout layer (50%) to prevent overfitting
   - Output layer with sigmoid activation (one neuron per step)

3. **Fine-Tuning Process**:
   - First, only the classification head is trained
   - Then, the last 30 layers of MobileNetV2 are unfrozen
   - Fine-tuning occurs with a lower learning rate 

### Sensor Model Architecture

The sensor system uses a sequential Long Short-Term Memory (LSTM) network designed to identify temporal patterns in sensor data:

1. **Input Layer**: Takes a window of sensor readings (default: 100 time steps)

2. **LSTM Stack**:
   - First LSTM layer: 64 units with return sequences enabled
   - Second LSTM layer: 32 units for context analysis

3. **Classification Head**:
   - Dense layer with 16 neurons and ReLU activation
   - Dropout layer (50%)
   - Output layer with sigmoid activation (one neuron per step)

## Configuration

Create a `config.json` file with the following structure:

```json
{
    "camera_sources": [0],
    "sensor_config": {
        "pressure": {"min": 0, "max": 200},
        "temperature": {"min": 0, "max": 100},
        "vibration": {"min": 0, "max": 10},
        "proximity": {"min": 0, "max": 100}
    },
    "step_definitions": [
        "Component placement",
        "Screw fastening",
        "Quality inspection"
    ],
    "model_paths": {
        "vision": "models/vision_model.h5",
        "sensor": "models/sensor_model.h5"
    }
}
```

## Socket.IO Events

### Client → Server

- `connect`: Client connects to the server
- `disconnect`: Client disconnects from the server
- `request_history`: Request full history data
- `reset_process`: Reset the task
- `send_alert`: Send an alert to be broadcast to all clients

### Server → Client

- `initial_data`: Sent on connection with current state and history
- `vision_update`: Real-time vision processing results
- `sensor_update`: Real-time sensor processing results
- `state_update`: Combined state analysis updates
- `frame_analysis`: Results from analyzing a specific frame
- `alert`: System alerts and notifications
- `history_update`: Updates to the system history
- `history_data`: Full history data response
- `system_status`: System status change notifications


## Dataset Structure

This dataset contains synthetic vision and sensor data for training machine learning models to task steps.

```
dataset/
├── vision/              # Computer vision data for step detection
│   ├── train/           # Training images (70%)
│   ├── val/             # Validation images (10%)
│   └── test/            # Testing images (20%)
├── sensor/              # Sensor time-series data
│   ├── train_sensor_data.csv  # Training sensor readings
│   ├── val_sensor_data.csv    # Validation sensor readings
│   └── test_sensor_data.csv   # Testing sensor readings
└── metadata.json        # Dataset configuration and metadata
```

### Vision Data

Each step folder contains synthetic 640x480 JPEG images showing assembly line products with visual state indicators, divided into:
- `complete/`: Step successfully completed
- `incomplete/`: Step in progress or not done

### Sensor Data

Time series readings from simulated sensors with columns:
- `timestamp`: Reading time
- `pressure`, `temperature`, `proximity`, `vibration`: Sensor values
- `step_id`: Current assembly step (0-4)
- `completed`: Completion flag (1=complete, 0=incomplete)

## Running the Code

1. **Generate the dataset** using the provided script:
   ```bash
   python app_cli.py generate --config config.json --output dataset --num-sensor-samples 10000 --num-images-per-step 100
   ```

2. **Prepare the dataset** by splitting into train/val/test sets:
   ```bash
   python app_cli.py prepare --config config.json --input dataset --output prepared_data
   ```

3. **Train ML models** on the prepared data:
   ```bash
   python app_cli.py train --data prepared_data --config config.json --epochs 50 --learning-rate 0.0005
   ```

4. **Quick test** with sample data:
   ```bash
   python app_cli.py test --config config.json --vision-model models/vision/vision_model.h5 --image dataset/vision/step_3_Cable_Connection/incomplete/0000.jpg
   ```

## Test the Sensor Model

To test just the sensor model on specific sensor data:

```bash
# Test on complete step sensor data
python tool_test_models.py \
  --config config.json \
  --sensor-model models/sensor_model.h5 \
  --sensor-data test_samples/sensor_data/Screw_Fastening_complete.csv \
  --scaler models/sensor_processed/sensor_scalers.json \
  --output test_results/sensor

# Test on incomplete step sensor data
python tool_test_models.py \
  --config config.json \
  --sensor-model models/sensor_model.h5 \
  --sensor-data test_samples/sensor_data/Cable_Connection_incomplete.csv \
  --scaler models/sensor_processed/sensor_scalers.json \
  --output test_results/sensor
```

## Test Both Models Together

To test both models together for integrated predictions:

```bash
# Test on complete step data
python tool_test_models.py \
  --config config.json \
  --vision-model models/vision_model.h5 \
  --image test_samples/images/Quality_Check_complete.jpg \
  --sensor-model models/sensor_model.h5 \
  --sensor-data test_samples/sensor_data/Quality_Check_complete.csv \
  --scaler models/sensor_processed/sensor_scalers.json \
  --output test_results/combined

# Test on incomplete step data
python tool_test_models.py \
  --config config.json \
  --vision-model models/vision_model.h5 \
  --image test_samples/images/Quality_Check_incomplete.jpg \
  --sensor-model models/sensor_model.h5 \
  --sensor-data test_samples/sensor_data/Quality_Check_incomplete.csv \
  --scaler models/sensor_processed/sensor_scalers.json \
  --output test_results/combined
```

## Testing Multiple Steps

To evaluate model performance across multiple steps:

```bash
# Create a batch testing script
for step in Component_Placement Circuit_Board_Mounting Screw_Fastening Cable_Connection Quality_Check; do
  echo "Testing $step (Complete)"
  python tool_test_models.py \
    --config config.json \
    --vision-model models/vision_model.h5 \
    --image test_samples/images/${step}_complete.jpg \
    --sensor-model models/sensor_model.h5 \
    --sensor-data test_samples/sensor_data/${step}_complete.csv \
    --scaler models/sensor_processed/sensor_scalers.json \
    --output test_results/${step}_complete
  
  echo "Testing $step (Incomplete)"
  python tool_test_models.py \
    --config config.json \
    --vision-model models/vision_model.h5 \
    --image test_samples/images/${step}_incomplete.jpg \
    --sensor-model models/sensor_model.h5 \
    --sensor-data test_samples/sensor_data/${step}_incomplete.csv \
    --scaler models/sensor_processed/sensor_scalers.json \
    --output test_results/${step}_incomplete
done
```