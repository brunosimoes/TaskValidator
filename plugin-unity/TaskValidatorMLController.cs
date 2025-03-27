using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;

public class TaskValidatorMLController : MonoBehaviour
{
  [Header("Model Configuration")]
  [SerializeField] private NNModel visionModelAsset;
  [SerializeField] private NNModel sensorModelAsset;
  [SerializeField] private TextAsset configFile;

  [Header("Camera Settings")]
  [SerializeField] private Camera detectionCamera;
  [SerializeField] private int captureWidth = 224;
  [SerializeField] private int captureHeight = 224;

  [Header("Sensor Input")]
  [SerializeField] private bool useSensorData = true;
  [SerializeField] private float[] pressureValues;
  [SerializeField] private float[] temperatureValues;
  [SerializeField] private float[] proximityValues;
  [SerializeField] private float[] vibrationValues;

  [Header("Detection Settings")]
  [SerializeField] private float detectionThreshold = 0.7f;
  [SerializeField] private float detectionInterval = 0.5f;

  [Header("Debug")]
  [SerializeField] private bool debugMode = true;
  [SerializeField] private bool visualizeResults = true;

  // Barracuda model variables
  private Model visionRuntimeModel;
  private Model sensorRuntimeModel;
  private IWorker visionWorker;
  private IWorker sensorWorker;

  // Step definitions from config
  private string[] stepDefinitions;
  private AssemblyLineConfig config;

  // Results
  private float[] visionResults;
  private float[] sensorResults;
  private bool[] completedSteps;

  // Detection state
  private bool isProcessing = false;
  private RenderTexture renderTexture;
  private Texture2D processTexture;

  void Start()
  {
    InitializeModels();
    InitializeConfig();
    InitializeDetection();

    // Start detection coroutine
    StartCoroutine(DetectionLoop());
  }

  void OnDestroy()
  {
    // Clean up workers
    visionWorker?.Dispose();
    sensorWorker?.Dispose();

    // Clean up textures
    if (renderTexture != null)
      Destroy(renderTexture);
    if (processTexture != null)
      Destroy(processTexture);
  }

  private void InitializeModels()
  {
    // Initialize vision model
    visionRuntimeModel = ModelLoader.Load(visionModelAsset);
    visionWorker = WorkerFactory.CreateWorker(visionRuntimeModel, WorkerFactory.Device.GPU);

    // Initialize sensor model if needed
    if (useSensorData)
    {
      sensorRuntimeModel = ModelLoader.Load(sensorModelAsset);
      sensorWorker = WorkerFactory.CreateWorker(sensorRuntimeModel, WorkerFactory.Device.CPU);
    }

    Debug.Log("ML models initialized");
  }

  private void InitializeConfig()
  {
    if (configFile != null)
    {
      config = JsonUtility.FromJson<AssemblyLineConfig>(configFile.text);
      stepDefinitions = config.step_definitions;

      // Initialize results arrays
      visionResults = new float[stepDefinitions.Length];
      sensorResults = new float[stepDefinitions.Length];
      completedSteps = new bool[stepDefinitions.Length];

      Debug.Log($"Loaded {stepDefinitions.Length} assembly steps from config");
    }
    else
    {
      Debug.LogError("Config file not assigned!");
    }
  }

  private void InitializeDetection()
  {
    if (detectionCamera == null)
    {
      detectionCamera = Camera.main;
      Debug.Log("Using main camera for detection");
    }

    // Create render texture for camera capture
    renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
    processTexture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
  }

  private IEnumerator DetectionLoop()
  {
    while (true)
    {
      if (!isProcessing)
      {
        isProcessing = true;

        // Process vision
        ProcessVision();

        // Process sensors if enabled
        if (useSensorData)
        {
          ProcessSensors();
        }

        // Integrate results
        IntegrateResults();

        // Visualize if needed
        if (visualizeResults)
        {
          VisualizeResults();
        }

        isProcessing = false;
      }

      yield return new WaitForSeconds(detectionInterval);
    }
  }

  private void ProcessVision()
  {
    // Capture camera view to texture
    detectionCamera.targetTexture = renderTexture;
    detectionCamera.Render();

    // Read pixels from render texture
    RenderTexture.active = renderTexture;
    processTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
    processTexture.Apply();
    RenderTexture.active = null;
    detectionCamera.targetTexture = null;

    // Convert to tensor
    using (var tensor = new Tensor(processTexture, channels: 3))
    {
      // Execute inference
      visionWorker.Execute(tensor);
      Tensor output = visionWorker.PeekOutput();

      // Get results (assuming output is step completion probabilities)
      for (int i = 0; i < visionResults.Length && i < output.length; i++)
      {
        visionResults[i] = output[i];
      }

      if (debugMode)
      {
        Debug.Log("Vision inference complete: " + string.Join(", ", visionResults.Select(v => v.ToString("F2"))));
      }
    }
  }

  private void ProcessSensors()
  {
    // Prepare sensor data tensor
    // Assuming the model expects a time series of sensor readings
    int timeSteps = pressureValues.Length;
    int numFeatures = 4; // pressure, temperature, proximity, vibration

    float[] sensorData = new float[timeSteps * numFeatures];

    for (int t = 0; t < timeSteps; t++)
    {
      sensorData[t * numFeatures + 0] = pressureValues[t];
      sensorData[t * numFeatures + 1] = temperatureValues[t];
      sensorData[t * numFeatures + 2] = proximityValues[t];
      sensorData[t * numFeatures + 3] = vibrationValues[t];
    }

    // Shape is [batch=1, time_steps, features]
    using (var tensor = new Tensor(1, timeSteps, numFeatures, sensorData))
    {
      // Execute inference
      sensorWorker.Execute(tensor);
      Tensor output = sensorWorker.PeekOutput();

      // Get results
      for (int i = 0; i < sensorResults.Length && i < output.length; i++)
      {
        sensorResults[i] = output[i];
      }

      if (debugMode)
      {
        Debug.Log("Sensor inference complete: " + string.Join(", ", sensorResults.Select(v => v.ToString("F2"))));
      }
    }
  }

  private void IntegrateResults()
  {
    // Combine vision and sensor results
    // A step is considered complete if both vision and sensor models agree (or just vision if sensors disabled)
    for (int i = 0; i < completedSteps.Length; i++)
    {
      if (useSensorData)
      {
        // Weighted average (vision 60%, sensors 40%)
        float combinedConfidence = 0.6f * visionResults[i] + 0.4f * sensorResults[i];
        completedSteps[i] = combinedConfidence > detectionThreshold;
      }
      else
      {
        completedSteps[i] = visionResults[i] > detectionThreshold;
      }
    }

    // Call event handlers with results
    OnStepDetectionComplete();
  }

  private void OnStepDetectionComplete()
  {
    // Log results
    if (debugMode)
    {
      for (int i = 0; i < completedSteps.Length; i++)
      {
        string status = completedSteps[i] ? "COMPLETE" : "INCOMPLETE";
        Debug.Log($"Step {i}: {stepDefinitions[i]} - {status}");
      }
    }

    // Fire events for game logic to handle
    StepDetectionEvent detectionEvent = new StepDetectionEvent
    {
      CompletedSteps = completedSteps,
      StepDefinitions = stepDefinitions,
      VisionConfidence = visionResults,
      SensorConfidence = sensorResults
    };

    // Notify other components about the detection results
    StepDetectionEventManager.TriggerEvent(detectionEvent);
  }

  private void VisualizeResults()
  {
    // This would be implemented to show visual feedback in the Unity scene
    // Could update UI elements, materials, or other visual indicators
  }

  // Method to update sensor values from external sources
  public void UpdateSensorValues(float[] pressure, float[] temperature, float[] proximity, float[] vibration)
  {
    pressureValues = pressure;
    temperatureValues = temperature;
    proximityValues = proximity;
    vibrationValues = vibration;
  }

  // For debug/testing: force detection update
  public void TriggerDetection()
  {
    if (!isProcessing)
    {
      StartCoroutine(ForcedDetection());
    }
  }

  private IEnumerator ForcedDetection()
  {
    isProcessing = true;
    ProcessVision();
    if (useSensorData) ProcessSensors();
    IntegrateResults();
    if (visualizeResults) VisualizeResults();
    isProcessing = false;
    yield return null;
  }
}

// Config class matching the JSON structure
[Serializable]
public class AssemblyLineConfig
{
  public string[] step_definitions;
  public SensorConfig sensor_config;
}

[Serializable]
public class SensorConfig
{
  public SensorRangeConfig pressure;
  public SensorRangeConfig temperature;
  public SensorRangeConfig proximity;
  public SensorRangeConfig vibration;
}

[Serializable]
public class SensorRangeConfig
{
  public float[] expected_range;
  public float alert_threshold;
}

// Event system for detection results
public class StepDetectionEvent
{
  public bool[] CompletedSteps;
  public string[] StepDefinitions;
  public float[] VisionConfidence;
  public float[] SensorConfidence;
}

// Simple event manager to decouple detection from UI/game logic
public static class StepDetectionEventManager
{
  public delegate void StepDetectionEventHandler(StepDetectionEvent e);
  public static event StepDetectionEventHandler OnStepDetection;

  public static void TriggerEvent(StepDetectionEvent e)
  {
    OnStepDetection?.Invoke(e);
  }
}