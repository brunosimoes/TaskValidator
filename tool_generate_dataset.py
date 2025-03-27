import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import math
import argparse
from tqdm import tqdm
import json

def generate_sensor_data(config, output_file, num_samples=10000):
    """
    Generate synthetic sensor data for task steps
    
    Parameters:
    config (dict): Configuration with sensor settings
    output_file (str): Path to save the CSV file
    num_samples (int): Number of samples to generate
    """
    print(f"Generating {num_samples} sensor data samples...")
    
    # Extract sensor configuration
    sensor_config = config["sensor_config"]
    step_definitions = config["step_definitions"]
    num_steps = len(step_definitions)
    
    # Create DataFrame
    columns = ["timestamp"] + list(sensor_config.keys()) + ["step_id", "completed"]
    df = pd.DataFrame(columns=columns)
    
    # Generate time series
    timestamp_base = pd.Timestamp.now()
    
    # Generate data for each step
    current_step = 0
    step_transition_prob = 0.005  # Probability to move to next step
    
    for i in tqdm(range(num_samples)):
        timestamp = timestamp_base + pd.Timedelta(seconds=i*0.1)
        
        # Determine if we move to the next step
        if (current_step < num_steps - 1) and (random.random() < step_transition_prob):
            current_step += 1
        
        # Generate sensor values based on the current step
        row = {"timestamp": timestamp, "step_id": current_step}
        
        # Each step has characteristic sensor patterns
        for sensor_name, sensor_info in sensor_config.items():
            expected_min, expected_max = sensor_info["expected_range"]
            
            # Base value depends on the current step
            base_value = expected_min + (expected_max - expected_min) * (0.3 + 0.7 * (current_step / (num_steps - 1)))
            
            # Add noise
            noise = np.random.normal(0, (expected_max - expected_min) * 0.1)
            
            # Add periodic component (simulates machine cycles)
            periodic = math.sin(i * 0.2) * (expected_max - expected_min) * 0.15
            
            # Final value
            value = base_value + noise + periodic
            
            # Clip to reasonable range
            value = max(0, min(value, sensor_info["alert_threshold"] * 1.2))
            
            row[sensor_name] = value
        
        # Mark as completed if we're past 80% of the typical cycle for this step
        # and add some randomness to completion status
        is_step_completed = random.random() < 0.8
        row["completed"] = is_step_completed
        
        # Add row to DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df['completed'] = df['completed'].astype(bool)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Sensor data saved to {output_file}")

def generate_image_data(config, output_dir, num_images_per_step=100, image_size=(640, 480)):
    """
    Generate synthetic images for task steps
    
    Parameters:
    config (dict): Configuration with step definitions
    output_dir (str): Directory to save the images
    num_images_per_step (int): Number of images to generate per step
    image_size (tuple): Image dimensions (width, height)
    """
    print(f"Generating {num_images_per_step} images per assembly step...")
    
    step_definitions = config["step_definitions"]
    
    # Create directories for each step
    for i, step_name in enumerate(step_definitions):
        step_dir = os.path.join(output_dir, f"step_{i}_{step_name.replace(' ', '_')}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Complete and incomplete subdirectories
        complete_dir = os.path.join(step_dir, "complete")
        incomplete_dir = os.path.join(step_dir, "incomplete")
        os.makedirs(complete_dir, exist_ok=True)
        os.makedirs(incomplete_dir, exist_ok=True)
        
        print(f"Generating images for Step {i}: {step_name}")
        
        # Generate images for this step
        for j in tqdm(range(num_images_per_step)):
            # Decide if this is a completed or incomplete example
            is_complete = random.random() < 0.5
            target_dir = complete_dir if is_complete else incomplete_dir
            
            # Create synthetic image
            img = create_image(i, step_definitions, is_complete, image_size)
            
            # Save the image
            img_path = os.path.join(target_dir, f"{j:04d}.jpg")
            cv2.imwrite(img_path, img)

def create_image(step_id, step_definitions, is_complete, image_size):
    """
    Create a synthetic image at a specific step
    
    Parameters:
    step_id (int): Current step ID
    step_definitions (list): List of step names
    is_complete (bool): Whether the step is completed
    image_size (tuple): Image dimensions
    
    Returns:
    numpy.ndarray: Synthetic image
    """
    width, height = image_size
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw a conveyor belt
    cv2.rectangle(img, (50, height-150), (width-50, height-100), (120, 120, 120), -1)
    
    # Draw tracks
    for i in range(5):
        y = height - 150 + i * 10
        cv2.line(img, (50, y), (width-50, y), (80, 80, 80), 2)
    
    # Draw a product being assembled
    product_width = 200
    product_height = 100
    product_x = (width - product_width) // 2
    product_y = height - 200
    
    # Base product
    cv2.rectangle(img, (product_x, product_y), (product_x + product_width, product_y + product_height), (200, 200, 200), -1)
    cv2.rectangle(img, (product_x, product_y), (product_x + product_width, product_y + product_height), (100, 100, 100), 2)
    
    # Different components based on step
    colors = [
        (0, 0, 255),    # Step 0 - Red component
        (0, 255, 0),    # Step 1 - Green component
        (255, 0, 0),    # Step 2 - Blue component
        (255, 255, 0),  # Step 3 - Yellow component
        (0, 255, 255)   # Step 4 - Cyan component
    ]
    
    # Draw components for completed steps
    for i in range(step_id):
        comp_x = product_x + 20 + i * 35
        comp_y = product_y + 20
        cv2.circle(img, (comp_x, comp_y), 15, colors[i], -1)
    
    # Draw current step component (partially if incomplete)
    if step_id < len(colors):
        comp_x = product_x + 20 + step_id * 35
        comp_y = product_y + 20
        
        if is_complete:
            cv2.circle(img, (comp_x, comp_y), 15, colors[step_id], -1)
        else:
            # Draw a hand/tool working on it
            cv2.circle(img, (comp_x, comp_y), 15, colors[step_id], 1)
            
            # Draw a simulated tool
            tool_x = comp_x + 20
            tool_y = comp_y - 20
            cv2.line(img, (tool_x, tool_y), (comp_x, comp_y), (50, 50, 50), 3)
            cv2.circle(img, (tool_x, tool_y), 8, (50, 50, 50), -1)
    
    # Add some noise to simulate camera variance
    noise = np.random.normal(0, 5, img.shape).astype(np.int8)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add lighting variation
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Add text label
    step_name = step_definitions[step_id]
    status = "COMPLETE" if is_complete else "IN PROGRESS"
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # Add simple text without requiring a font
    text = f"Step {step_id}: {step_name} - {status}"
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic task dataset')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', required=True, help='Output directory for dataset')
    parser.add_argument('--num_sensor_samples', type=int, default=10000, help='Number of sensor data samples')
    parser.add_argument('--num_images_per_step', type=int, default=100, help='Number of images per step')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sensor data
    sensor_output = os.path.join(args.output_dir, 'sensor_data.csv')
    generate_sensor_data(config, sensor_output, args.num_sensor_samples)
    
    # Generate image data
    image_output_dir = os.path.join(args.output_dir, 'vision')
    generate_image_data(config, image_output_dir, args.num_images_per_step)
    
    print("Dataset generation complete!")

if __name__ == '__main__':
    main()