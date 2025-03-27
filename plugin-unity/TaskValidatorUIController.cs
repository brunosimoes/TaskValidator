using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class TaskValidatorUIController : MonoBehaviour
{
  [Header("UI References")]
  [SerializeField] private RectTransform stepsContainer;
  [SerializeField] private GameObject stepPrefab;
  [SerializeField] private TextMeshProUGUI statusText;
  [SerializeField] private Image statusIcon;
  [SerializeField] private TextMeshProUGUI completionText;

  [Header("UI Colors")]
  [SerializeField] private Color completeColor = new Color(0.2f, 0.8f, 0.2f);
  [SerializeField] private Color incompleteColor = new Color(0.8f, 0.2f, 0.2f);
  [SerializeField] private Color inProgressColor = new Color(0.2f, 0.6f, 0.9f);

  [Header("Icons")]
  [SerializeField] private Sprite completeIcon;
  [SerializeField] private Sprite incompleteIcon;
  [SerializeField] private Sprite inProgressIcon;

  // Track UI elements for steps
  private List<StepUIElements> stepUIElements = new List<StepUIElements>();

  // Current state
  private bool[] completedSteps;
  private string[] stepDefinitions;
  private float[] confidenceValues;

  private void Start()
  {
    // Register for detection events
    StepDetectionEventManager.OnStepDetection += HandleStepDetection;
  }

  private void OnDestroy()
  {
    // Unregister from events
    StepDetectionEventManager.OnStepDetection -= HandleStepDetection;
  }

  private void HandleStepDetection(StepDetectionEvent e)
  {
    // Store current state
    completedSteps = e.CompletedSteps;
    stepDefinitions = e.StepDefinitions;

    // Combine vision and sensor confidence
    confidenceValues = new float[e.VisionConfidence.Length];
    for (int i = 0; i < confidenceValues.Length; i++)
    {
      confidenceValues[i] = (e.VisionConfidence[i] + e.SensorConfidence[i]) / 2;
    }

    // Initialize UI if needed
    if (stepUIElements.Count == 0 && stepDefinitions != null)
    {
      InitializeStepUI();
    }

    // Update UI with detection results
    UpdateStepUI();

    // Update overall status
    UpdateStatusUI();
  }

  private void InitializeStepUI()
  {
    // Clear any existing elements
    foreach (Transform child in stepsContainer)
    {
      Destroy(child.gameObject);
    }
    stepUIElements.Clear();

    // Create UI elements for each step
    for (int i = 0; i < stepDefinitions.Length; i++)
    {
      GameObject stepObj = Instantiate(stepPrefab, stepsContainer);
      StepUIElements elements = new StepUIElements
      {
        Container = stepObj.GetComponent<RectTransform>(),
        StepLabel = stepObj.transform.Find("StepLabel").GetComponent<TextMeshProUGUI>(),
        StepIcon = stepObj.transform.Find("StatusIcon").GetComponent<Image>(),
        ProgressBar = stepObj.transform.Find("ProgressBar").GetComponent<Slider>(),
        ConfidenceText = stepObj.transform.Find("ConfidenceText").GetComponent<TextMeshProUGUI>()
      };

      // Set the step name
      elements.StepLabel.text = $"Step {i + 1}: {stepDefinitions[i]}";

      stepUIElements.Add(elements);
    }
  }

  private void UpdateStepUI()
  {
    for (int i = 0; i < stepUIElements.Count; i++)
    {
      if (i < completedSteps.Length)
      {
        StepUIElements ui = stepUIElements[i];
        bool isCompleted = completedSteps[i];
        float confidence = confidenceValues[i];

        // Update icon
        ui.StepIcon.sprite = isCompleted ? completeIcon : incompleteIcon;
        ui.StepIcon.color = isCompleted ? completeColor : incompleteColor;

        // Update progress bar
        ui.ProgressBar.value = confidence;

        // Set the progress bar color based on confidence
        Image progressFill = ui.ProgressBar.fillRect.GetComponent<Image>();
        if (progressFill != null)
        {
          if (confidence < 0.4f)
            progressFill.color = incompleteColor;
          else if (confidence < 0.7f)
            progressFill.color = inProgressColor;
          else
            progressFill.color = completeColor;
        }

        // Update confidence text
        ui.ConfidenceText.text = $"{confidence * 100:F0}%";
      }
    }
  }

  private void UpdateStatusUI()
  {
    if (completedSteps == null || completedSteps.Length == 0)
      return;

    // Calculate overall completion
    int completedCount = 0;
    for (int i = 0; i < completedSteps.Length; i++)
    {
      if (completedSteps[i])
        completedCount++;
    }

    // Determine current step
    int currentStep = -1;
    for (int i = 0; i < completedSteps.Length; i++)
    {
      if (!completedSteps[i])
      {
        currentStep = i;
        break;
      }
    }

    // All steps completed
    if (completedCount == completedSteps.Length)
    {
      statusText.text = "Assembly Complete";
      statusIcon.sprite = completeIcon;
      statusIcon.color = completeColor;
      completionText.text = $"All steps complete: {completedCount}/{completedSteps.Length}";
    }
    // No steps completed
    else if (completedCount == 0)
    {
      statusText.text = "Assembly Not Started";
      statusIcon.sprite = incompleteIcon;
      statusIcon.color = incompleteColor;
      completionText.text = $"No steps complete: {completedCount}/{completedSteps.Length}";
    }
    // In progress
    else
    {
      string stepName = currentStep >= 0 && currentStep < stepDefinitions.Length ?
                        stepDefinitions[currentStep] : "Unknown";

      statusText.text = $"In Progress: {stepName}";
      statusIcon.sprite = inProgressIcon;
      statusIcon.color = inProgressColor;
      completionText.text = $"Steps complete: {completedCount}/{completedSteps.Length}";
    }
  }

  // Helper class to group UI elements for a step
  private class StepUIElements
  {
    public RectTransform Container;
    public TextMeshProUGUI StepLabel;
    public Image StepIcon;
    public Slider ProgressBar;
    public TextMeshProUGUI ConfidenceText;
  }
}