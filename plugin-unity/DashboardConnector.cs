using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class DashboardConnector : MonoBehaviour
{
  [Header("Server Configuration")]
  [SerializeField] private string serverUrl = "http://localhost:5000";
  [SerializeField] private float updateInterval = 1.0f;
  [SerializeField] private bool sendUpdatesAutomatically = true;

  [Header("Components")]
  [SerializeField] private TaskValidatorMLController mlController;

  // Connection status
  private bool isConnected = false;
  private string connectionStatus = "Not connected";

  // Last detection event
  private StepDetectionEvent lastEvent;

  private void Start()
  {
    // Find ML controller if not assigned
    if (mlController == null)
    {
      mlController = FindObjectOfType<TaskValidatorMLController>();
    }

    // Register for detection events
    StepDetectionEventManager.OnStepDetection += HandleStepDetection;

    // Start connection check
    StartCoroutine(CheckConnection());

    // Start automatic updates if enabled
    if (sendUpdatesAutomatically)
    {
      StartCoroutine(SendPeriodicUpdates());
    }
  }

  private void OnDestroy()
  {
    // Unregister from events
    StepDetectionEventManager.OnStepDetection -= HandleStepDetection;
  }

  private void HandleStepDetection(StepDetectionEvent e)
  {
    // Store the last event
    lastEvent = e;

    // Send update immediately if not using periodic updates
    if (!sendUpdatesAutomatically)
    {
      StartCoroutine(SendStatusUpdate(e));
    }
  }

  private IEnumerator CheckConnection()
  {
    while (true)
    {
      UnityWebRequest request = UnityWebRequest.Get($"{serverUrl}/api/status");
      yield return request.SendWebRequest();

      if (request.result == UnityWebRequest.Result.Success)
      {
        isConnected = true;
        connectionStatus = "Connected";
        Debug.Log("Connected to dashboard server");
      }
      else
      {
        isConnected = false;
        connectionStatus = $"Connection failed: {request.error}";
        Debug.LogWarning($"Dashboard connection failed: {request.error}");
      }

      yield return new WaitForSeconds(5.0f);
    }
  }

  private IEnumerator SendPeriodicUpdates()
  {
    while (true)
    {
      if (isConnected && lastEvent != null)
      {
        yield return SendStatusUpdate(lastEvent);
      }

      yield return new WaitForSeconds(updateInterval);
    }
  }

  private IEnumerator SendStatusUpdate(StepDetectionEvent e)
  {
    if (!isConnected)
    {
      Debug.LogWarning("Not connected to dashboard server, update not sent");
      yield break;
    }

    // Create JSON data
    var updateData = new Dictionary<string, object>
        {
            { "completedSteps", e.CompletedSteps },
            { "stepDefinitions", e.StepDefinitions },
            { "visionConfidence", e.VisionConfidence },
            { "sensorConfidence", e.SensorConfidence },
            { "timestamp", DateTime.UtcNow.ToString("o") }
        };

    string jsonData = JsonUtility.ToJson(new SerializableDict<string, object>(updateData));

    // Send the data
    using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/api/update", "POST"))
    {
      byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
      request.uploadHandler = new UploadHandlerRaw(bodyRaw);
      request.downloadHandler = new DownloadHandlerBuffer();
      request.SetRequestHeader("Content-Type", "application/json");

      yield return request.SendWebRequest();

      if (request.result == UnityWebRequest.Result.Success)
      {
        Debug.Log("Status update sent to dashboard");
      }
      else
      {
        Debug.LogError($"Failed to send status update: {request.error}");
      }
    }
  }

  // Send a manual alert to the dashboard
  public void SendAlert(string alertType, string message)
  {
    if (!isConnected)
    {
      Debug.LogWarning("Not connected to dashboard server, alert not sent");
      return;
    }

    StartCoroutine(SendAlertCoroutine(alertType, message));
  }

  private IEnumerator SendAlertCoroutine(string alertType, string message)
  {
    var alertData = new Dictionary<string, string>
        {
            { "type", alertType },
            { "message", message }
        };

    string jsonData = JsonUtility.ToJson(new SerializableDict<string, string>(alertData));

    using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/api/alert", "POST"))
    {
      byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
      request.uploadHandler = new UploadHandlerRaw(bodyRaw);
      request.downloadHandler = new DownloadHandlerBuffer();
      request.SetRequestHeader("Content-Type", "application/json");

      yield return request.SendWebRequest();

      if (request.result == UnityWebRequest.Result.Success)
      {
        Debug.Log("Alert sent to dashboard");
      }
      else
      {
        Debug.LogError($"Failed to send alert: {request.error}");
      }
    }
  }

  // Reset the assembly status on the dashboard
  public void ResetAssemblyStatus()
  {
    if (!isConnected)
    {
      Debug.LogWarning("Not connected to dashboard server, reset not sent");
      return;
    }

    StartCoroutine(ResetAssemblyStatusCoroutine());
  }

  private IEnumerator ResetAssemblyStatusCoroutine()
  {
    using (UnityWebRequest request = new UnityWebRequest($"{serverUrl}/api/reset", "POST"))
    {
      request.downloadHandler = new DownloadHandlerBuffer();
      request.SetRequestHeader("Content-Type", "application/json");

      yield return request.SendWebRequest();

      if (request.result == UnityWebRequest.Result.Success)
      {
        Debug.Log("Assembly status reset on dashboard");
      }
      else
      {
        Debug.LogError($"Failed to reset assembly status: {request.error}");
      }
    }
  }

  // Helper class to make dictionaries serializable for JSON
  [Serializable]
  private class SerializableDict<TKey, TValue>
  {
    [SerializeField]
    private List<TKey> keys = new List<TKey>();

    [SerializeField]
    private List<TValue> values = new List<TValue>();

    public SerializableDict(Dictionary<TKey, TValue> dictionary)
    {
      foreach (var kvp in dictionary)
      {
        keys.Add(kvp.Key);
        values.Add(kvp.Value);
      }
    }

    public Dictionary<TKey, TValue> ToDictionary()
    {
      var dict = new Dictionary<TKey, TValue>();

      for (int i = 0; i < Mathf.Min(keys.Count, values.Count); i++)
      {
        dict[keys[i]] = values[i];
      }

      return dict;
    }
  }
}