# Task Monitoring System

A Flask and Socket.IO based application for monitoring task completition using computer vision and sensor data analysis in real-time.

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