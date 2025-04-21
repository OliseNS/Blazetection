// DOM Elements
const fireToggle = document.getElementById('fire-toggle');
const smokeToggle = document.getElementById('smoke-toggle');
const motionToggle = document.getElementById('motion-toggle');
const faceToggle = document.getElementById('face-toggle');
const telegramToggle = document.getElementById('telegram-toggle');
const telegramToken = document.getElementById('telegram-token');
const telegramChatId = document.getElementById('telegram-chat-id');
const telegramCooldown = document.getElementById('telegram-cooldown');
const saveTelegramBtn = document.getElementById('save-telegram-btn');
const testTelegramBtn = document.getElementById('test-telegram-btn');
const stopAlarmBtn = document.getElementById('stop-alarm-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusIcon = document.getElementById('status-icon');
const statusText = document.getElementById('status-text');
const facesGallery = document.getElementById('faces-gallery');
const faceModal = document.getElementById('face-modal');
const modalImage = document.getElementById('modal-image');
const closeModal = document.getElementById('close-modal');
const ttsSpeakBtn = document.getElementById('tts-speak-btn');
const ttsTextInput = document.getElementById('tts-text');
const motionStatus = document.getElementById('motionStatus');
const videoFeed = document.getElementById('live-feed');
const frameCount = document.getElementById('frames-processed');
const statusSummary = document.getElementById('current-status');
const detectionGallery = document.getElementById('detection-gallery');

// Configuration and Status
let config = {};
let systemStatus = 'Normal';

// Array to store TTS history
let ttsHistory = [];

// Debounce function to limit function calls for better performance
function debounce(func, wait) {
    let timeout;
    return function () {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

// Handle fullscreen functionality for video feed
function setupVideoFullscreen() {
    const fullscreenBtn = document.getElementById('fullscreen-btn');

    function toggleFullScreen() {
        if (!document.fullscreenElement) {
            if (videoFeed.requestFullscreen) {
                videoFeed.requestFullscreen();
            } else if (videoFeed.webkitRequestFullscreen) {
                videoFeed.webkitRequestFullscreen();
            } else if (videoFeed.msRequestFullscreen) {
                videoFeed.msRequestFullscreen();
            }
            fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
            fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
        }
    }

    if (videoFeed && fullscreenBtn) {
        videoFeed.addEventListener('click', toggleFullScreen);
        fullscreenBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleFullScreen();
        });

        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement) {
                fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
            }
        });
    }
}

// Fetch system statistics
function fetchStatistics() {
    fetch('/api/statistics')
        .then((response) => response.json())
        .then((data) => {
            // Update only the numeric parts of the statistics
            const uptimeElement = document.getElementById('uptime');
            const framesProcessedElement = document.getElementById('frames-processed');

            if (uptimeElement) {
                const uptimeText = `Uptime: ${data.uptime}`;
                if (uptimeElement.textContent !== uptimeText) {
                    uptimeElement.textContent = uptimeText;
                }
            }

            if (framesProcessedElement) {
                const framesText = `Frames: ${data.frames_processed.toLocaleString()}`;
                if (framesProcessedElement.textContent !== framesText) {
                    framesProcessedElement.textContent = framesText;
                }
            }
        })
        .catch((error) => console.error('Error fetching statistics:', error));
}

// Update UI with statistics
function updateStatisticsUI(data) {
    document.getElementById('uptime').textContent = data.uptime;
    frameCount.textContent = data.frames_processed.toLocaleString();
    statusSummary.textContent = data.status;
}

// Initialize the dashboard
function initDashboard() {
    fetchConfig();
    fetchStatus();
    fetchFaces();
    fetchDetections();
    setupEventListeners();
    setupVideoFullscreen();
    fetchStatistics();

    setInterval(fetchStatus, 3000);
    setInterval(fetchFaces, 10000);
    setInterval(fetchDetections, 10000);
    setInterval(fetchStatistics, 5000);
}

// Fetch configuration from server
function fetchConfig() {
    fetch('/api/config')
        .then((response) => response.json())
        .then((data) => {
            config = data;
            updateConfigUI();
        })
        .catch((error) => console.error('Error fetching config:', error));
}

// Update UI based on configuration
function updateConfigUI() {
    fireToggle.checked = config.detection?.fire ?? true;
    smokeToggle.checked = config.detection?.smoke ?? true;
    motionToggle.checked = config.detection?.motion ?? true;
    faceToggle.checked = config.detection?.face ?? true;

    telegramToggle.checked = config.telegram?.enabled ?? false;
    telegramToken.value = config.telegram?.token ?? '';
    telegramChatId.value = config.telegram?.chat_id ?? '';
    telegramCooldown.value = config.telegram?.cooldown ?? 30;
}

// Fetch system status from server
function fetchStatus() {
    fetch('/api/status')
        .then((response) => response.json())
        .then((data) => {
            updateStatusUI(data);
        })
        .catch((error) => console.error('Error fetching status:', error));
}

// Update UI based on system status
function updateStatusUI(data) {
    systemStatus = data.status;
    statusText.textContent = systemStatus;
    statusIndicator.className = '';

    switch (systemStatus.toLowerCase()) {
        case 'fire detected':
        case 'fire pre-alert':
            statusIndicator.classList.add('status-fire');
            statusIcon.innerHTML = '<i class="fas fa-fire"></i>';
            break;
        case 'smoke detected':
        case 'smoke pre-alert':
            statusIndicator.classList.add('status-smoke');
            statusIcon.innerHTML = '<i class="fas fa-smog"></i>';
            break;
        case 'motion detected':
            motionStatus.querySelector('.status-dot').style.backgroundColor = 'green';
            motionStatus.querySelector('.last-detection-time').textContent = new Date().toLocaleTimeString();
            break;
        case 'person detected':
        case 'face detected':
            statusIndicator.classList.add('status-person');
            statusIcon.innerHTML = '<i class="fas fa-user"></i>';
            break;
        default:
            statusIndicator.classList.add('status-normal');
            statusIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
    }
}

// Improved fetch faces function to handle different response formats
function fetchFaces() {
    fetch('/api/faces')
        .then((response) => response.json())
        .then((data) => {
            // Check if data is in expected format
            let faces = [];
            if (data.faces && Array.isArray(data.faces)) {
                faces = data.faces;
            } else if (Array.isArray(data)) {
                faces = data;
            }
            
            // Debug log
            console.log('Faces data:', faces);
            
            updateFaceGallery(faces);
        })
        .catch((error) => {
            console.error('Error fetching faces:', error);
            document.getElementById('faces-gallery').innerHTML = 
                '<div class="no-faces">Error loading faces</div>';
        });
}

// Simplified face gallery update - modify to work with filenames
function updateFaceGallery(faces) {
    const gallery = document.getElementById('faces-gallery');
    
    if (!faces || faces.length === 0) {
        gallery.innerHTML = '<div class="no-faces">No faces detected yet</div>';
        return;
    }
    
    gallery.innerHTML = '';
    faces.forEach((face) => {
        const faceTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const faceItem = document.createElement('div');
        faceItem.className = 'face-item';
        
        // Construct the image URL from the filename
        const imageUrl = `/faces/${face}`;
        
        faceItem.innerHTML = `
            <img src="${imageUrl}" alt="Face">
            <div class="face-timestamp">${faceTime}</div>
        `;
        
        // Add click event for modal
        faceItem.addEventListener('click', () => {
            const modal = document.getElementById('face-modal');
            const modalImg = document.getElementById('modal-image');
            modal.style.display = 'flex';
            modalImg.src = imageUrl;
        });
        
        gallery.appendChild(faceItem);
    });
}

// Fetch recent detections from the server
function fetchDetections() {
    fetch('/api/detections')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data && Array.isArray(data.detections)) {
                updateDetectionGallery(data.detections);
            } else {
                console.warn('Invalid detections data format:', data);
                updateDetectionGallery([]);
            }
        })
        .catch(error => {
            console.error('Error fetching detections:', error);
            if (detectionGallery) {
                detectionGallery.innerHTML = '<div class="no-detections">Error loading detections</div>';
            }
        });
}

// Simplified detection gallery - only show time, fix image path
function updateDetectionGallery(detections) {
    const gallery = document.getElementById('detection-gallery');
    
    if (!detections || detections.length === 0) {
        gallery.innerHTML = '<div class="no-detections">No recent detections</div>';
        return;
    }
    
    gallery.innerHTML = '';
    detections.forEach(filename => {
        // Get current time for display
        const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item';
        
        // Construct proper image URL from filename
        const imageUrl = `/detections/${filename}`;
        
        detectionItem.innerHTML = `
            <img src="${imageUrl}" alt="Detection">
            <div class="detection-timestamp">${currentTime}</div>
        `;
        
        // Add click event for modal
        detectionItem.addEventListener('click', () => {
            openDetectionModal(imageUrl);
        });
        
        gallery.appendChild(detectionItem);
    });
}

// Helper function to format time only
function formatTime() {
    const now = new Date();
    let hours = now.getHours();
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    
    hours = hours % 12;
    hours = hours ? hours : 12; // Convert 0 to 12 for 12 AM
    
    return `${hours}:${minutes} ${ampm}`;
}

// Open modal with expanded face image
function openModal(imageSrc) {
    modalImage.src = imageSrc;
    faceModal.style.display = 'flex';
}

// Open modal with expanded detection image
function openDetectionModal(imageSrc) {
    console.log('Opening detection modal with image:', imageSrc);
    const modalImage = document.getElementById('modal-image');
    const faceModal = document.getElementById('face-modal');
    
    if (!modalImage || !faceModal) {
        console.error('Modal elements not found!');
        return;
    }
    
    modalImage.src = imageSrc;
    faceModal.style.display = 'flex';
}

// Close modal
function closeModalFunc() {
    const faceModal = document.getElementById('face-modal');
    if (faceModal) {
        faceModal.style.display = 'none';
    }
}

// Ensure event listeners for modal close functionality are set up
document.addEventListener('DOMContentLoaded', () => {
    const closeModal = document.getElementById('close-modal');
    const faceModal = document.getElementById('face-modal');

    if (closeModal) {
        closeModal.addEventListener('click', closeModalFunc);
    }

    if (faceModal) {
        window.addEventListener('click', (event) => {
            if (event.target === faceModal) closeModalFunc();
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && faceModal.style.display === 'flex') closeModalFunc();
        });
    }
});

// Set up event listeners for UI interactions
function setupEventListeners() {
    fireToggle.addEventListener('change', () => updateDetectionConfig('fire', fireToggle.checked));
    smokeToggle.addEventListener('change', () => updateDetectionConfig('smoke', smokeToggle.checked));
    motionToggle.addEventListener('change', () => updateDetectionConfig('motion', motionToggle.checked));
    faceToggle.addEventListener('change', () => updateDetectionConfig('face', faceToggle.checked));

    saveTelegramBtn.addEventListener('click', saveTelegramSettings);
    testTelegramBtn.addEventListener('click', testTelegramConnection);
    stopAlarmBtn.addEventListener('click', stopAlarm);
    closeModal.addEventListener('click', closeModalFunc);

    window.addEventListener('click', (event) => {
        if (event.target === faceModal) closeModalFunc();
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && faceModal.style.display === 'flex') closeModalFunc();
    });

    ttsSpeakBtn.addEventListener('click', () => {
        const text = ttsTextInput.value.trim();
        if (!text) {
            showNotification('Please enter some text.', 'warning');
            return;
        }

        fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.success) {
                    showNotification('Text is being spoken on the Raspberry Pi.', 'success');

                    // Add text to history only if it's not already present
                    if (!ttsHistory.includes(text)) {
                        ttsHistory.unshift(text); // Add to the beginning
                        if (ttsHistory.length > 5) {
                            ttsHistory.pop(); // Remove the oldest item
                        }
                    }

                    // Update the TTS history UI
                    updateTTSHistoryUI();
                } else {
                    showNotification('Failed to process TTS.', 'error');
                }
            })
            .catch(() => showNotification('Error with TTS. Please try again.', 'error'));
    });
}

// Update detection configuration
function updateDetectionConfig(type, enabled) {
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section: 'detection', values: { [type]: enabled } }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (!data.success) {
                showNotification('Failed to update detection settings', 'error');
            }
        })
        .catch(() => showNotification('Error updating detection settings', 'error'));
}

// Save Telegram settings
function saveTelegramSettings() {
    const values = {
        enabled: telegramToggle.checked,
        token: telegramToken.value,
        chat_id: telegramChatId.value,
        cooldown: parseInt(telegramCooldown.value) || 30,
    };

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section: 'telegram', values }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.success) {
                showNotification('Telegram settings saved successfully!', 'success');
            } else {
                showNotification('Failed to save Telegram settings.', 'error');
            }
        })
        .catch(() => showNotification('Error saving Telegram settings.', 'error'));
}

// Test Telegram connection
function testTelegramConnection() {
    fetch('/api/telegram/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.success) {
                showNotification('Test message sent successfully!', 'success');
            } else {
                showNotification('Failed to send test message.', 'error');
            }
        })
        .catch(() => showNotification('Error testing Telegram connection.', 'error'));
}

// Stop alarm
function stopAlarm() {
    fetch('/api/alarm/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.success) {
                showNotification('Alarm stopped successfully', 'success');
            } else {
                showNotification('Failed to stop alarm.', 'error');
            }
        })
        .catch(() => showNotification('Error stopping alarm.', 'error'));
}

// Show notification
function showNotification(message, type = 'info') {
    let notification = document.getElementById('notification');
    if (notification) {
        notification.parentNode.removeChild(notification);
    }

    notification = document.createElement('div');
    notification.id = 'notification';

    switch (type) {
        case 'success':
            notification.style.backgroundColor = 'var(--success-color)';
            break;
        case 'error':
            notification.style.backgroundColor = 'var(--danger-color)';
            break;
        case 'warning':
            notification.style.backgroundColor = 'var(--warning-color)';
            notification.style.color = '#333';
            break;
        default:
            notification.style.backgroundColor = 'var(--primary-color)';
    }

    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

// Function to update TTS history UI
function updateTTSHistoryUI() {
    const ttsHistoryContainer = document.getElementById('tts-history');
    ttsHistoryContainer.innerHTML = ''; // Clear existing history

    if (ttsHistory.length === 0) {
        ttsHistoryContainer.innerHTML = '<div class="no-faces">No TTS history yet</div>';
        return;
    }

    ttsHistory.forEach((text, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = 'tts-history-item';
        historyItem.textContent = text;

        // Add click event to replay the text
        historyItem.addEventListener('click', () => {
            fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.success) {
                        showNotification(`Replaying: "${text}"`, 'success');
                    } else {
                        showNotification('Failed to replay TTS.', 'error');
                    }
                })
                .catch(() => showNotification('Error replaying TTS. Please try again.', 'error'));
        });

        ttsHistoryContainer.appendChild(historyItem);
    });
}

// Refresh header stats
function refreshHeaderStats() {
    setInterval(() => {
        fetch('/api/statistics')
            .then((response) => response.json())
            .then((data) => {
                document.getElementById('uptime').textContent = `Uptime: ${data.uptime}`;
                document.getElementById('frames-processed').textContent = `Frames: ${data.frames_processed}`;
            })
            .catch((error) => console.error('Error refreshing header stats:', error));
    }, 500); // Refresh every 0.5 seconds
}

// Reload Telegram configuration
function reloadTelegramConfig() {
    fetch('/api/telegram/reload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Telegram configuration reloaded!', 'success');
        } else {
            showNotification('Failed to reload Telegram configuration.', 'error');
        }
    })
    .catch(() => showNotification('Error reloading Telegram configuration.', 'error'));
}

// Add event listener for reload button
document.addEventListener('DOMContentLoaded', () => {
    const reloadTelegramBtn = document.getElementById('reload-telegram-btn');
    if (reloadTelegramBtn) {
        reloadTelegramBtn.addEventListener('click', reloadTelegramConfig);
    }
});

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    refreshHeaderStats();
});
