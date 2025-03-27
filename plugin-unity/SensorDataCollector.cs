using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SensorDataCollector : MonoBehaviour
{
  [Header("Sensor Configuration")]
  [SerializeField] private int bufferSize = 100;
  [SerializeField] private float sampleRate = 0.1f; // seconds
  [SerializeField] private bool simulateSensors = true;

  [Header("Sensor Simulation Settings")]
  [SerializeField] private AnimationCurve pressureSimulation;
  [SerializeField] private AnimationCurve temperatureSimulation;
  [SerializeField] private AnimationCurve proximitySimulation;
  [SerializeField] private AnimationCurve vibrationSimulation;
  [SerializeField] private float noiseAmount = 0.1f;

  [Header("External References")]
  [SerializeField] private TaskValidatorMLController mlController;

  // Circular buffers for sensor data
  private float[] pressureBuffer;
  private float[] temperatureBuffer;
  private float[] proximityBuffer;
  private float[] vibrationBuffer;

  // Buffer index
  private int currentIndex = 0;
  private bool bufferFilled = false;

  // Simulation time
  private float simulationTime = 0f;

  void Start()
  {
    // Initialize buffers
    pressureBuffer = new float[bufferSize];
    temperatureBuffer = new float[bufferSize];
    proximityBuffer = new float[bufferSize];
    vibrationBuffer = new float[bufferSize];

    // Find ML controller if not assigned
    if (mlController == null)
    {
      mlController = FindObjectOfType<TaskValidatorMLController>();
    }

    // Start data collection
    StartCoroutine(CollectSensorData());
  }

  private IEnumerator CollectSensorData()
  {
    while (true)
    {
      // Collect sensor data
      if (simulateSensors)
      {
        CollectSimulatedData();
      }
      else
      {
        CollectRealData();
      }

      // Update buffer index
      currentIndex = (currentIndex + 1) % bufferSize;
      if (currentIndex == 0)
      {
        bufferFilled = true;
      }

      // Update ML controller with latest buffer
      if (mlController != null && bufferFilled)
      {
        mlController.UpdateSensorValues(
            pressureBuffer,
            temperatureBuffer,
            proximityBuffer,
            vibrationBuffer
        );
      }

      yield return new WaitForSeconds(sampleRate);
    }
  }

  private void CollectSimulatedData()
  {
    // Increment simulation time
    simulationTime += sampleRate;

    // Get base values from animation curves (0-1 range)
    float pressureBase = pressureSimulation.Evaluate(simulationTime % pressureSimulation.keys[pressureSimulation.length - 1].time);
    float temperatureBase = temperatureSimulation.Evaluate(simulationTime % temperatureSimulation.keys[temperatureSimulation.length - 1].time);
    float proximityBase = proximitySimulation.Evaluate(simulationTime % proximitySimulation.keys[proximitySimulation.length - 1].time);
    float vibrationBase = vibrationSimulation.Evaluate(simulationTime % vibrationSimulation.keys[vibrationSimulation.length - 1].time);

    // Add noise
    float pressureNoise = UnityEngine.Random.Range(-noiseAmount, noiseAmount);
    float temperatureNoise = UnityEngine.Random.Range(-noiseAmount, noiseAmount);
    float proximityNoise = UnityEngine.Random.Range(-noiseAmount, noiseAmount);
    float vibrationNoise = UnityEngine.Random.Range(-noiseAmount, noiseAmount);

    // Scale to realistic ranges
    pressureBuffer[currentIndex] = ScaleValue(pressureBase + pressureNoise, 0, 1, 90, 110);
    temperatureBuffer[currentIndex] = ScaleValue(temperatureBase + temperatureNoise, 0, 1, 20, 30);
    proximityBuffer[currentIndex] = ScaleValue(proximityBase + proximityNoise, 0, 1, 40, 60);
    vibrationBuffer[currentIndex] = ScaleValue(vibrationBase + vibrationNoise, 0, 1, 1, 3);
  }

  private void CollectRealData()
  {
    // In a real application, this would interface with actual hardware sensors
    // via serial port, network, or device-specific APIs

    // Example with mock hardware interface:
    // pressureBuffer[currentIndex] = hardwareInterface.GetPressureReading();
    // temperatureBuffer[currentIndex] = hardwareInterface.GetTemperatureReading();
    // proximityBuffer[currentIndex] = hardwareInterface.GetProximityReading();
    // vibrationBuffer[currentIndex] = hardwareInterface.GetVibrationReading();

    Debug.LogWarning("Real sensor data collection not implemented");

    // Use simulated data as fallback
    CollectSimulatedData();
  }

  private float ScaleValue(float value, float oldMin, float oldMax, float newMin, float newMax)
  {
    // Clamp to old range
    value = Mathf.Clamp(value, oldMin, oldMax);

    // Scale to new range
    return ((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin;
  }

  // Method to connect to custom hardware (could be called from UI or other scripts)
  public bool ConnectToHardware(string connectionString)
  {
    // Implementation would depend on the specific hardware
    // For example, connecting to an Arduino over serial:
    // serialPort.Open(connectionString, baudRate);

    Debug.Log($"Connecting to hardware: {connectionString}");

    // Return success status
    return false; // Placeholder
  }

  // Method to get the latest sensor values (for debugging or UI)
  public SensorSnapshot GetLatestSensorValues()
  {
    if (!bufferFilled && currentIndex == 0)
      return null;

    int index = currentIndex > 0 ? currentIndex - 1 : bufferSize - 1;

    return new SensorSnapshot
    {
      pressure = pressureBuffer[index],
      temperature = temperatureBuffer[index],
      proximity = proximityBuffer[index],
      vibration = vibrationBuffer[index],
      timestamp = Time.time
    };
  }

  // Method to reset simulation
  public void ResetSimulation()
  {
    simulationTime = 0;

    // Clear buffers
    for (int i = 0; i < bufferSize; i++)
    {
      pressureBuffer[i] = 0;
      temperatureBuffer[i] = 0;
      proximityBuffer[i] = 0;
      vibrationBuffer[i] = 0;
    }

    currentIndex = 0;
    bufferFilled = false;
  }
}

// Structure to hold a snapshot of sensor values
[Serializable]
public struct SensorSnapshot
{
  public float pressure;
  public float temperature;
  public float proximity;
  public float vibration;
  public float timestamp;
}