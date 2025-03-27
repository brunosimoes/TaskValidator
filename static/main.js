const serverUrl = 'http://localhost:5000';
let socket;
let assemblyData = {
  currentState: {
    completedSteps: [],
    stepDefinitions: [],
    stepConfidence: {},
    allCompleted: false,
    timestamp: null
  },
  history: [],
  alerts: []
};
let historyData = [];
let selectedImage = null;

// DOM Elements
const stepsContainer = document.getElementById('stepsContainer');
const alertsContainer = document.getElementById('alertsContainer');
const systemStatus = document.getElementById('systemStatus');
const connectionStatus = document.getElementById('connectionStatus');
const statusIndicator = document.getElementById('statusIndicator');
const historyContainer = document.getElementById('historyContainer');
const fullHistoryContainer = document.getElementById('fullHistoryContainer');
const resetButton = document.getElementById('resetButton');

// Image Analysis DOM Elements
const fileInput = document.getElementById('fileInput');
const browseButton = document.getElementById('browseButton');
const dropZone = document.getElementById('dropZone');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');
const clearImageButton = document.getElementById('clearImageButton');
const analyzeButton = document.getElementById('analyzeButton');
const analysisResults = document.getElementById('analysisResults');
const analysisLoadingIndicator = document.getElementById('analysisLoadingIndicator');

function init() {
  connectSocketIO();
  setupEventListeners();
  fetchInitialData();
  setupImageUploadHandlers();
}

function connectSocketIO() {
  socket = io(serverUrl, {
    withCredentials: true,
    transports: ['websocket']
  });

  socket.on('connect', () => {
    console.log('Socket.IO connected');
    connectionStatus.textContent = 'Connected';
    statusIndicator.classList.remove('bg-red-500', 'bg-yellow-500');
    statusIndicator.classList.add('bg-green-500');
  });

  socket.on('disconnect', () => {
    console.log('Socket.IO disconnected');
    connectionStatus.textContent = 'Disconnected';
    statusIndicator.classList.remove('bg-green-500', 'bg-yellow-500');
    statusIndicator.classList.add('bg-red-500');
  });

  socket.on('connect_error', (error) => {
    console.error('Socket.IO connection error:', error);
    connectionStatus.textContent = 'Connection Error';
    statusIndicator.classList.remove('bg-green-500', 'bg-red-500');
    statusIndicator.classList.add('bg-yellow-500');
  });

  socket.on('initial_data', (data) => {
    console.log('Received initial data:', data);
    assemblyData = {
      ...assemblyData,
      ...data.data,
      alerts: Array.isArray(data.data.alerts) ? data.data.alerts : [],
      history: Array.isArray(data.data.history) ? data.data.history : []
    };
    updateUI();
  });

  socket.on('state_update', (data) => {
    assemblyData.currentState = data.data;
    updateStepsUI();
    updateSystemStatusUI();
  });

  socket.on('vision_update', (data) => {
    console.log('Vision update from camera', data.camera);
  });

  socket.on('sensor_update', (data) => {
    console.log('Sensor update received');
  });

  socket.on('alert', (data) => {
    if (!assemblyData.alerts) {
      assemblyData.alerts = [];
    }
    assemblyData.alerts.unshift(data.data);
    if (assemblyData.alerts.length > 20) {
      assemblyData.alerts = assemblyData.alerts.slice(0, 20);
    }
    updateAlertsUI();
  });

  socket.on('history_data', (data) => {
    historyData = data.data;
    updateFullHistoryUI();
  });

  socket.on('history_update', (data) => {
    if (!Array.isArray(historyData)) {
      historyData = [];
    }

    if (!Array.isArray(assemblyData.history)) {
      assemblyData.history = [];
    }

    if (historyData.length > 0) {
      historyData.unshift(data.data);
    }

    assemblyData.history.unshift(data.data);
    if (assemblyData.history.length > 10) {
      assemblyData.history = assemblyData.history.slice(0, 10);
    }
    updateHistoryUI();
  });

  socket.on('frame_analysis', (data) => {
    console.log('Received frame analysis:', data);
  });

  socket.on('system_status', (data) => {
    console.log('System status changed:', data);

    if (data.message) {
      addAlert({
        type: data.status === 'error' ? 'error' : 'success',
        message: data.message,
        timestamp: data.timestamp
      });
    }

    if (data.status === 'reset') {
      assemblyData.currentState = {
        completedSteps: [],
        stepDefinitions: assemblyData.currentState.stepDefinitions,
        stepConfidence: {},
        allCompleted: false,
        timestamp: data.timestamp
      };
      updateStepsUI();
      updateSystemStatusUI();
    }
  });

  socket.on('server_message', (data) => {
    console.log('Server message:', data);
    if (data.message) {
      addAlert({
        type: data.type || 'info',
        message: data.message,
        timestamp: new Date().getTime()
      });
    }
  });
}

function setupEventListeners() {
  resetButton.addEventListener('click', resetAssembly);
}

function fetchInitialData() {
  fetch(`${serverUrl}/api/status`)
    .then(response => response.json())
    .then(data => {
      assemblyData.currentState = data;
      updateStepsUI();
      updateSystemStatusUI();
    })
    .catch(error => {
      console.error('Error fetching status:', error);
    });

  fetch(`${serverUrl}/api/alerts`)
    .then(response => response.json())
    .then(data => {
      assemblyData.alerts = Array.isArray(data) ? data : [];
      updateAlertsUI();
    })
    .catch(error => {
      console.error('Error fetching alerts:', error);
    });

  fetch(`${serverUrl}/api/history?limit=10`)
    .then(response => response.json())
    .then(data => {
      assemblyData.history = Array.isArray(data) ? data : [];
      updateHistoryUI();
    })
    .catch(error => {
      console.error('Error fetching history:', error);
    });
}

function updateUI() {
  updateStepsUI();
  updateAlertsUI();
  updateHistoryUI();
  updateSystemStatusUI();
}

function updateStepsUI() {
  if (!assemblyData.currentState.stepDefinitions ||
    assemblyData.currentState.stepDefinitions.length === 0) {
    stepsContainer.innerHTML = '<div class="text-gray-500">No step data available</div>';
    return;
  }

  stepsContainer.innerHTML = '';

  assemblyData.currentState.stepDefinitions.forEach((step, index) => {
    const isCompleted = assemblyData.currentState.completedSteps.includes(index);

    let confidence = 0;
    if (assemblyData.currentState.stepConfidence &&
      assemblyData.currentState.stepConfidence[step]) {
      confidence = assemblyData.currentState.stepConfidence[step].combined || 0;
    }

    const stepHTML = `
            <div class="border rounded-md p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        ${isCompleted ?
        '<svg class="w-6 h-6 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>' :
        '<svg class="w-6 h-6 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>'
      }
                        <span class="font-medium">${step}</span>
                    </div>
                    <div class="px-3 py-1 rounded-full text-sm ${getConfidenceColorClass(confidence)}">
                        ${Math.round(confidence * 100)}% confidence
                    </div>
                </div>
                <div class="mt-2 bg-gray-200 rounded-full h-2.5">
                    <div 
                        class="h-2.5 rounded-full ${isCompleted ? 'bg-green-500' : 'bg-blue-500'}"
                        style="width: ${confidence * 100}%"
                    ></div>
                </div>
            </div>
        `;

    stepsContainer.innerHTML += stepHTML;
  });
}

function updateAlertsUI() {
  if (!assemblyData?.alerts || assemblyData.alerts.length === 0) {
    alertsContainer.innerHTML = '<div class="text-gray-500">No alerts at this time</div>';
    return;
  }

  alertsContainer.innerHTML = '';

  assemblyData.alerts.forEach(alert => {
    const alertType = alert.type || 'info';
    const alertColor = getAlertColorClass(alertType);
    const timestamp = moment(alert.timestamp).fromNow();

    const alertHTML = `
            <div class="p-3 rounded-md flex items-start ${alertColor}">
                <svg class="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                </svg>
                <div>
                    <p class="font-medium">${alert.message}</p>
                    <p class="text-sm opacity-75">${timestamp}</p>
                </div>
            </div>
        `;

    alertsContainer.innerHTML += alertHTML;
  });
}

function getConfidenceColorClass(confidence) {
  if (confidence > 0.8) return 'bg-green-200 text-green-800';
  if (confidence > 0.5) return 'bg-yellow-200 text-yellow-800';
  return 'bg-red-200 text-red-800';
}

function getAlertColorClass(type) {
  switch (type) {
    case 'error':
      return 'bg-red-100 text-red-800';
    case 'warning':
      return 'bg-yellow-100 text-yellow-800';
    case 'success':
      return 'bg-green-100 text-green-800';
    case 'lowConfidence':
      return 'bg-orange-100 text-orange-800';
    default:
      return 'bg-blue-100 text-blue-800';
  }
}

function setupImageUploadHandlers() {
  fileInput.addEventListener('change', handleFileInputChange);
  
  browseButton.addEventListener('click', () => {
    fileInput.click();
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500', 'bg-blue-50');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500', 'bg-blue-50');
  });

  dropZone.addEventListener('drop', handleFileDrop);
  clearImageButton.addEventListener('click', clearSelectedImage);
  analyzeButton.addEventListener('click', analyzeSelectedImage);
}

function handleFileInputChange(event) {
  const file = event.target.files[0];
  if (file) {
    processSelectedFile(file);
  }
}

function handleFileDrop(event) {
  event.preventDefault();
  dropZone.classList.remove('border-blue-500', 'bg-blue-50');

  if (event.dataTransfer.files.length) {
    const file = event.dataTransfer.files[0];
    processSelectedFile(file);
  }
}

function processSelectedFile(file) {
  if (!file.type.startsWith('image/')) {
    addAlert({
      type: 'error',
      message: 'The selected file is not an image.',
      timestamp: new Date()
    });
    return;
  }

  selectedImage = file;
  const objectUrl = URL.createObjectURL(file);
  imagePreview.src = objectUrl;

  dropZone.classList.add('hidden');
  previewContainer.classList.remove('hidden');

  analyzeButton.disabled = false;
  analyzeButton.classList.remove('opacity-50', 'cursor-not-allowed');
}

function clearSelectedImage() {
  fileInput.value = '';
  selectedImage = null;

  if (imagePreview.src.startsWith('blob:')) {
    URL.revokeObjectURL(imagePreview.src);
  }

  imagePreview.src = '';
  previewContainer.classList.add('hidden');
  dropZone.classList.remove('hidden');

  analyzeButton.disabled = true;
  analyzeButton.classList.add('opacity-50', 'cursor-not-allowed');

  analysisResults.innerHTML = '<div class="text-gray-500 text-center py-8">Upload an image to see analysis results</div>';
}

function analyzeSelectedImage() {
  if (!selectedImage) {
    return;
  }

  analysisResults.classList.add('hidden');
  analysisLoadingIndicator.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', selectedImage);

  fetch(`${serverUrl}/analyze-frame`, {
    method: 'POST',
    body: formData
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      displayAnalysisResults(data);
    })
    .catch(error => {
      console.error('Error analyzing image:', error);
      displayAnalysisError(error.message);
    })
    .finally(() => {
      analysisLoadingIndicator.classList.add('hidden');
      analysisResults.classList.remove('hidden');
    });
}

function displayAnalysisResults(data) {
  if (data.status !== 'success') {
    displayAnalysisError(data.message || 'Unknown error occurred');
    return;
  }

  const results = data.results;

  function mapResultsToAssemblyData() {
    const completedSteps = results.completed_steps;
    const stepDefinitions = Object.values(results.step_details).map(step => step.description);

    const stepConfidence = {};
    Object.entries(results.step_details).forEach(([stepId, details]) => {
      stepConfidence[details.description] = {
        combined: details.confidence_score
      };
    });

    return {
      completedSteps: completedSteps,
      stepDefinitions: stepDefinitions,
      stepConfidence: stepConfidence,
      allCompleted: results.all_completed,
      timestamp: data.results.timestamp * 1000
    };
  }

  assemblyData.currentState = mapResultsToAssemblyData();

  updateStepsUI();
  updateSystemStatusUI();

  let resultsHTML = '<div class="space-y-4">';

  if (results.completed_steps) {
    resultsHTML += `
      <div class="p-3 rounded-md bg-blue-50 border border-blue-200">
        <h4 class="font-medium text-blue-700 mb-2">Detected Steps</h4>
        <ul class="space-y-1">
    `;

    if (results.completed_steps.length === 0) {
      resultsHTML += '<li class="text-blue-600">No steps detected in this image</li>';
    } else {
      results.completed_steps.forEach(stepId => {
        const stepName = results.step_details[stepId]?.description || `Step ${stepId}`;
        const confidence = results.step_details[stepId]?.confidence_score || 0;

        resultsHTML += `
          <li class="flex justify-between">
            <span class="text-blue-600">${stepName}</span>
            <span class="px-2 py-0.5 rounded-full text-xs ${getConfidenceColorClass(confidence)}">
              ${Math.round(confidence * 100)}%
            </span>
          </li>
        `;
      });
    }

    resultsHTML += '</ul></div>';
  }

  resultsHTML += `
    <div class="p-3 rounded-md ${results.all_completed ? 'bg-green-50 border border-green-200' : 'bg-yellow-50 border border-yellow-200'}">
      <h4 class="font-medium ${results.all_completed ? 'text-green-700' : 'text-yellow-700'} mb-2">Process Status</h4>
      <p class="${results.all_completed ? 'text-green-600' : 'text-yellow-600'}">
        ${results.all_completed ? 'All steps completed!' : 'Process incomplete - some steps not detected'}
      </p>
    </div>
  `;

  if (data.timestamp) {
    const formattedTime = moment(data.timestamp * 1000).format('YYYY-MM-DD HH:mm:ss');
    resultsHTML += `
      <div class="text-xs text-gray-500 text-right">
        Analyzed at: ${formattedTime}
      </div>
    `;
  }

  resultsHTML += '</div>';
  analysisResults.innerHTML = resultsHTML;
}

function displayAnalysisError(errorMessage) {
  analysisResults.innerHTML = `
    <div class="p-4 rounded-md bg-red-50 border border-red-200 text-center">
      <svg class="h-10 w-10 text-red-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
      </svg>
      <h4 class="font-medium text-red-700 mb-2">Analysis Failed</h4>
      <p class="text-red-600">${errorMessage}</p>
    </div>
  `;

  addAlert({
    type: 'error',
    message: `Image analysis failed: ${errorMessage}`,
    timestamp: new Date()
  });
}

function addAlert(alert) {
  if (!Array.isArray(assemblyData?.alerts)) {
    assemblyData.alerts = [];
  }

  assemblyData.alerts.unshift(alert);

  if (assemblyData.alerts.length > 20) {
    assemblyData.alerts = assemblyData.alerts.slice(0, 20);
  }

  updateAlertsUI();

  if (socket && socket.connected) {
    socket.emit('send_alert', {
      type: alert.type || 'info',
      message: alert.message,
      timestamp: alert.timestamp || new Date().getTime()
    });
  }
}

function updateHistoryUI() {
  if (!assemblyData?.history || assemblyData.history.length === 0) {
    historyContainer.innerHTML = `
      <tr>
        <td colspan="4" class="px-6 py-4 text-center text-gray-500">
          No history data available
        </td>
      </tr>
    `;
    return;
  }

  historyContainer.innerHTML = '';
  const recentHistory = assemblyData.history.slice(0, 10);

  recentHistory.forEach(entry => {
    const row = createHistoryRow(entry);
    historyContainer.innerHTML += row;
  });
}

function updateFullHistoryUI() {
  if (!historyData || historyData.length === 0) {
    fullHistoryContainer.innerHTML = `
      <tr>
        <td colspan="4" class="px-6 py-4 text-center text-gray-500">
          No history data available
        </td>
      </tr>
    `;
    return;
  }

  fullHistoryContainer.innerHTML = '';

  historyData.forEach(entry => {
    const row = createHistoryRow(entry);
    fullHistoryContainer.innerHTML += row;
  });
}

function createHistoryRow(entry) {
  let time = moment(entry.completedAt || entry.timestamp).format('YYYY-MM-DD HH:mm:ss');
  let event = entry.event || (entry.step !== undefined ? 'Step Completed' : 'Unknown');
  let details = entry.details || entry.stepName || (entry.event === 'reset' ? 'Assembly Reset' : '');
  let confidence = entry.confidence !== undefined ?
    `<span class="${getConfidenceColorClass(entry.confidence)}">${Math.round(entry.confidence * 100)}%</span>` :
    '';

  return `
    <tr>
      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        ${time}
      </td>
      <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
        ${event}
      </td>
      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        ${details}
      </td>
      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        ${confidence}
      </td>
    </tr>
  `;
}

function updateSystemStatusUI() {
  const lastUpdate = assemblyData.currentState.timestamp ?
    moment(assemblyData.currentState.timestamp).fromNow() :
    'Never';

  const completedCount = assemblyData.currentState.completedSteps ?
    assemblyData.currentState.completedSteps.length : 0;

  const totalSteps = assemblyData.currentState.stepDefinitions ?
    assemblyData.currentState.stepDefinitions.length : 0;

  systemStatus.innerHTML = `
    <div class="flex justify-between items-center">
      <span>ML Engine:</span>
      <span class="px-2 py-1 bg-green-100 text-green-800 rounded-md text-sm">Online</span>
    </div>
    <div class="flex justify-between items-center">
      <span>Last Update:</span>
      <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-md text-sm">${lastUpdate}</span>
    </div>
    <div class="flex justify-between items-center">
      <span>Completed Steps:</span>
      <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-md text-sm">${completedCount}/${totalSteps}</span>
    </div>
    <div class="flex justify-between items-center">
      <span>Status:</span>
      <span class="px-2 py-1 ${assemblyData.currentState.allCompleted ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'} rounded-md text-sm">
        ${assemblyData.currentState.allCompleted ? 'Complete' : 'In Progress'}
      </span>
    </div>
  `;
}

function resetAssembly() {
  if (socket && socket.connected) {
    socket.emit('reset_process');
  } else {
    fetch(`${serverUrl}/api/reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })
      .then(response => response.json())
      .then(data => {
        console.log('Reset successful:', data);

        const resetEvent = {
          event: 'reset',
          details: 'Task reset',
          timestamp: new Date().getTime()
        };

        if (assemblyData.history) {
          assemblyData.history.unshift(resetEvent);

          if (assemblyData.history.length > 10) {
            assemblyData.history = assemblyData.history.slice(0, 10);
          }

          updateHistoryUI();
        }

        addAlert({
          type: 'success',
          message: 'Task has been reset successfully',
          timestamp: new Date()
        });
      })
      .catch(error => {
        console.error('Error resetting assembly:', error);

        addAlert({
          type: 'error',
          message: `Error resetting assembly: ${error.message}`,
          timestamp: new Date()
        });
      });
  }
}

function checkConnectionStatus() {
  fetch(`${serverUrl}/api/status`, { timeout: 3000 })
    .then(response => {
      if (!response.ok) {
        throw new Error('Server returned error status');
      }
      return response.json();
    })
    .then(data => {
      connectionStatus.textContent = 'Connected';
      statusIndicator.classList.remove('bg-red-500', 'bg-yellow-500');
      statusIndicator.classList.add('bg-green-500');
    })
    .catch(error => {
      console.error('Connection error:', error);
      connectionStatus.textContent = 'Demo Mode';
      statusIndicator.classList.remove('bg-green-500', 'bg-red-500');
      statusIndicator.classList.add('bg-yellow-500');

      loadMockStepData();
    });
}

function loadMockStepData() {
  const mockStepDefinitions = [
    "Position component",
    "Apply adhesive",
    "Place top cover",
    "Secure with fasteners",
    "Apply quality check label"
  ];

  const mockCompletedSteps = [0, 1, 2];

  const mockConfidence = {};
  mockStepDefinitions.forEach((step, index) => {
    mockConfidence[step] = {
      combined: index < 3 ? 0.95 : 0.2
    };
  });

  assemblyData.currentState = {
    completedSteps: mockCompletedSteps,
    stepDefinitions: mockStepDefinitions,
    stepConfidence: mockConfidence,
    allCompleted: false,
    timestamp: new Date().getTime()
  };

  updateStepsUI();
  updateSystemStatusUI();

  addAlert({
    type: 'info',
    message: 'Using mock data for demonstration',
    timestamp: new Date()
  });

  return assemblyData.currentState;
}

document.addEventListener('DOMContentLoaded', () => {
  init();
  setTimeout(checkConnectionStatus, 1000);
});