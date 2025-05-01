import threading
import cv2
import os
import time
import json
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import euclidean
from datetime import datetime
from telegram import InputFile
from telegram.ext import ApplicationBuilder
import asyncio
import subprocess

try:
    from gpiozero import DigitalInputDevice  # Import gpiozero for MQ2 sensor
    GPIO_AVAILABLE = True
except ImportError:
    print("GPIO library not available. MQ2 sensor functionality will be disabled.")
    GPIO_AVAILABLE = False

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'

# Configuration management
class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.callbacks = []
        
    def _load_config(self):
        """Load configuration from JSON file or create default if not exists"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default configuration
                default_config = {
                    "detection": {
                        "fire": True,
                        "smoke": True,
                        "motion": True, 
                        "face": True
                    },
                    "telegram": {
                        "enabled": False,
                        "token": "",
                        "chat_id": "",
                        "cooldown": 30,
                    },
                    "system": {
                        "camera_index": 0,
                        "detection_interval": 0.5,
                        "face_save_interval": 1.0,
                        "alarm_threshold": 3,
                        "max_saved_faces": 50
                    }
                }
                self._save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {}
    
    def _save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False
    
    def get_config(self):
        """Get current configuration"""
        return self.config
    
    def update_config(self, new_config):
        """Update configuration and save to file"""
        self.config = new_config
        success = self._save_config()
        if success:
            self._notify_callbacks()
        return success
    
    def update_section(self, section, values):
        """Update a specific section of the configuration"""
        if section in self.config:
            self.config[section].update(values)
            success = self._save_config()
            if success:
                self._notify_callbacks()
            return success
        return False
    
    def register_callback(self, callback):
        """Register a callback function to be called when config changes"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks about config changes"""
        for callback in self.callbacks:
            try:
                callback(self.config)
            except Exception as e:
                print(f"Error in config callback: {str(e)}")
    
    def is_detection_enabled(self, detection_type):
        """Check if a specific detection type is enabled"""
        try:
            return self.config["detection"].get(detection_type, False)
        except:
            return False


class TelegramService:
    """
    Service for sending notifications and images to Telegram.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = self.config_manager.get_config()
        self.token = self.config["telegram"]["token"]
        self.chat_id = self.config["telegram"]["chat_id"]
        self.cooldown = self.config["telegram"]["cooldown"]
        self.last_notification_time = {}  # Track last notification time by type
        self.application = None
        self.bot = None
        self.loop = asyncio.get_event_loop()
        self.setup_bot()

    def reload_config(self):
        """Reload configuration and update bot if needed."""
        old_config = self.config
        self.config = self.config_manager.get_config()
        if old_config["telegram"]["token"] != self.config["telegram"]["token"]:
            self.token = self.config["telegram"]["token"]
            self.chat_id = self.config["telegram"]["chat_id"]
            self.cooldown = self.config["telegram"]["cooldown"]
            self.setup_bot()

    def setup_bot(self):
        """Initialize Telegram bot with current configuration."""
        if not self.config["telegram"]["enabled"] or not self.token:
            self.bot = None
            self.application = None
            return

        try:
            self.application = (
                ApplicationBuilder()
                .token(self.token)
                .concurrent_updates(True)  # Enable concurrent handling of updates
                .build()
            )
            self.bot = self.application.bot
            print(f"Telegram bot initialized with token: {self.token[:5]}...")
        except Exception as e:
            print(f"Error initializing Telegram bot: {e}")
            self.bot = None
            self.application = None

    def is_enabled(self):
        """Check if Telegram notifications are enabled."""
        return (
            self.config["telegram"]["enabled"]
            and self.bot is not None
            and self.chat_id is not None
        )

    def can_send_notification(self, notification_type):
        """
        Check if a notification can be sent based on a cooldown period.
        """
        if not self.is_enabled():
            print("Telegram notifications not enabled")
            return False

        current_time = time.time()
        if (notification_type not in self.last_notification_time or 
                (current_time - self.last_notification_time[notification_type]) >= self.cooldown):
            return True

        remaining = self.cooldown - (current_time - self.last_notification_time[notification_type])
        print(f"Telegram notification on cooldown for type: {notification_type}. Next notification in {remaining:.1f} seconds")
        return False

    async def send_notification_async(self, message, image_path=None, notification_type="general"):
        """
        Asynchronously send a notification message (and optionally an image) to Telegram.
        """
        if not self.is_enabled():
            print("Telegram notifications not enabled")
            return False

        if not self.can_send_notification(notification_type):
            return False

        # Update last notification time for this type
        self.last_notification_time[notification_type] = time.time()

        try:
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=InputFile(image_file),
                        caption=message,
                    )
                print(f"Telegram image sent: {image_path}")
            else:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                )
                print("Telegram message sent without image")

            print(f"Telegram notification sent: {notification_type}")
            return True
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
            return False

    def send_notification(self, message, image_path=None, notification_type="general"):
        """
        Synchronous wrapper for send_notification_async. Ensures that the event loop is active.
        """
        if self.loop.is_closed():
            print("Event loop is closed. Creating a new event loop.")
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        try:
            result = self.loop.run_until_complete(
                self.send_notification_async(message, image_path, notification_type)
            )
            return result
        except Exception as e:
            print(f"Exception in send_notification: {e}")
            return False

    def send_fire_alert(self, image_path=None):
        """Send a fire detection alert."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"🔥 FIRE DETECTED at {timestamp} - Check livestream: http://firepi.local:8080/"
        return self.send_notification(message, image_path, "fire")

    def send_smoke_alert(self, image_path=None):
        """Send a smoke detection alert."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"💨 SMOKE DETECTED at {timestamp} - Check livestream: http://firepi.local:8080/"
        return self.send_notification(message, image_path, "smoke")

    def send_motion_alert(self, image_path=None):
        """Send a person detection alert."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"👤 MOTION DETECTED at {timestamp} - Check livestream: http://firepi.local:8080/"
        return self.send_notification(message, image_path, "Motion")

    def send_face_alert(self, image_path=None):
        """Send a face detection alert."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"👁️ FACE DETECTED at {timestamp} - Check livestream: http://firepi.local:8080/"
        return self.send_notification(message, image_path, "face")

    def send_test_message(self):
        """Send a test message to verify Telegram configuration."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"✅ TEST: Telegram integration working at {timestamp} - Livestream: http://firepi.local:8080/"
        return self.send_notification(message, None, "test")

    def send_welcome_message(self):
        """Send a welcome message when the system starts."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"🔧 SYSTEM ONLINE at {timestamp} - Livestream available: http://firepi.local:8080/"
        return self.send_notification(message, None, "test")

    def send_gas_alert(self):
        """Send a gas detection alert."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"⚠️ GAS DETECTED at {timestamp} - Check your environment immediately! - http://firepi.local:8080/"
        return self.send_notification(message, None, "gas")

# Camera handling
class Camera:
    def __init__(self, source=None):
        """
        Initialize camera capture.
        If a source is provided, it can be a streaming URL or a numeric camera index.
        """
        if source is None:
            source = 0
        
        self.source = source
        self.cap = None
        self.connect()
        self.consecutive_failures = 0
        self.max_failures = 5
        self.reconnect_delay = 2  # seconds
        
    def connect(self):
        """Establish connection to camera"""
        try:
            if self.cap is not None:
                self.cap.release()
                
            # For IP cameras, set additional parameters for better reliability
            self.cap = cv2.VideoCapture(self.source)
            
            # If it's a string (URL), apply specific settings for better reliability
            if isinstance(self.source, str) and ("http://" in self.source or "rtsp://" in self.source):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) 
                
                # Set additional parameters for MJPEG streams
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                print(f"{Colors.CYAN}Connected to IP camera stream: {self.source}{Colors.RESET}")
            
            if not self.cap.isOpened():
                raise IOError(f"Error accessing camera source: {self.source}")
                
            print(f"{Colors.GREEN}Camera initialized successfully with source: {self.source}{Colors.RESET}")
            self.consecutive_failures = 0
        except Exception as e:
            print(f"{Colors.RED}Failed to connect to camera: {str(e)}{Colors.RESET}")
            raise
    
    def read_frame(self):
        """Read a frame from the camera with better error handling"""
        if self.cap is None or not self.cap.isOpened():
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                print(f"{Colors.YELLOW}Attempting to reconnect to camera...{Colors.RESET}")
                time.sleep(self.reconnect_delay)
                try:
                    self.connect()
                except:
                    pass
            return False, None
        
        try:
            success, frame = self.cap.read()
            
            if success:
                self.consecutive_failures = 0
                return success, frame
            else:
                self.consecutive_failures += 1
                
                # If we have multiple consecutive failures, attempt to reconnect
                if self.consecutive_failures >= self.max_failures:
                    print(f"{Colors.YELLOW}Multiple frame read failures. Attempting to reconnect...{Colors.RESET}")
                    time.sleep(self.reconnect_delay)
                    try:
                        self.connect()
                    except:
                        pass
                
                return False, None
        except Exception as e:
            print(f"{Colors.RED}Error reading frame: {str(e)}{Colors.RESET}")
            self.consecutive_failures += 1
            
            # If we have multiple consecutive failures, attempt to reconnect
            if self.consecutive_failures >= self.max_failures:
                print(f"{Colors.YELLOW}Error persists. Attempting to reconnect...{Colors.RESET}")
                time.sleep(self.reconnect_delay)
                try:
                    self.connect()
                except:
                    pass
                
            return False, None

    def release(self):
        """Release the camera resource"""
        if self.cap:
            self.cap.release()
            self.cap = None

# YOLO detector
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.65, iou_threshold=0.55):
        """Initialize YOLO object detector"""
        self.model = YOLO(model_path)
        self.CONF_THRESHOLD = conf_threshold
        self.IOU_THRESHOLD = iou_threshold
    
    def detect(self, frame):
        """Detect objects in frame using YOLO"""
        height, width = frame.shape[:2]
        results = self.model.predict(frame, imgsz=width, conf=self.CONF_THRESHOLD, iou=self.IOU_THRESHOLD)
        return results[0].boxes

# Face detector
class FaceDetector:
    def __init__(self, min_detection_confidence=0.8):
        """
        Initialize MediaPipe Face Mesh for more accurate and fast face detection.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,  # Adjust based on your use case
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def detect(self, frame_rgb):
        """
        Detect faces using MediaPipe Face Mesh.
        Returns a list of face landmarks if faces are detected.
        """
        results = self.face_mesh.process(frame_rgb)
        return results.multi_face_landmarks if results.multi_face_landmarks else []

# Motion detector
class MotionDetector:
    def __init__(self):
        """Initialize motion detector with better memory and performance"""
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
        self.min_area = 1000  # Minimum area to trigger motion detection
        self.last_motion_time = 0
        self.motion_cooldown = 2.0  # Time in seconds before motion is considered gone
        
    def detect(self, frame):
        """Detect motion in frame with automatic cooldown"""
        current_time = time.time()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)
        
        
        thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
     
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                motion_detected = True
                self.last_motion_time = current_time
                break
                
 
        return motion_detected or (current_time - self.last_motion_time < self.motion_cooldown)

class AudioPlayer:
    def __init__(self, sound_file="alarm.wav"):
        """Initialize audio player using aplay for Raspberry Pi"""
        self.sound_file = sound_file
        self.process = None
        self.playing = False
        
        # Verify sound file exists
        if not os.path.exists(self.sound_file):
            print(f"{Colors.YELLOW}Warning: Sound file {self.sound_file} not found. Alarm will be silent.{Colors.RESET}")
    
    def play_sound(self):
        """Play sound using aplay in a non-blocking way"""
        if self.playing:
            return  # Already playing
            
        try:
            if os.path.exists(self.sound_file):
                # Start aplay process
                self.process = subprocess.Popen(
                    ["aplay", self.sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.playing = True
            else:
                print(f"{Colors.YELLOW}Cannot play sound: {self.sound_file} not found{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error playing sound: {str(e)}{Colors.RESET}")
    
    def stop_sound(self):
        """Stop the currently playing sound"""
        if self.process and self.playing:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                # Force kill if terminate doesn't work
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
                self.playing = False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_sound()

# MQ2 Gas Sensor Handler
class MQ2Sensor:
    def __init__(self, pin=17):
        """
        Initialize MQ2 sensor on the specified GPIO pin.
        If GPIO is not available or running on a non-Raspberry Pi system, simulate the sensor.
        """
        if GPIO_AVAILABLE:
            try:
                self.sensor = DigitalInputDevice(pin)
            except Exception as e:
                print(f"GPIO initialization failed: {e}. Simulating sensor.")
                self.sensor = None
        else:
            self.sensor = None  # Simulate sensor
        self.gas_detected = False

    def read_sensor(self):
        """
        Read the MQ2 sensor value.
        Returns True if gas is detected, False otherwise.
        If GPIO is not available, always return False.
        """
        if self.sensor:
            # LOW signal indicates gas presence
            self.gas_detected = self.sensor.value == 0
        else:
            # Simulate no gas detected
            self.gas_detected = False
        return self.gas_detected

# Main detection system
class DetectionSystem:
    def __init__(self, config_path="config.json"):
        # Clean up old face images on startup
        self.cleanup_old_images()
        
        # Create output directories
        os.makedirs("faces", exist_ok=True)
        os.makedirs("detections", exist_ok=True)
        
        
        time.sleep(2)  
        
        # Initialize configuration and load the system config
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Use streaming URL if provided, otherwise fall back to the local camera index.
        camera_source = self.config["system"].get("camera_url")
        if camera_source:
            print(f"Using streaming URL as camera source: {camera_source}")
            self.camera = Camera(camera_source)
        else:
            camera_index = self.config["system"]["camera_index"]
            self.camera = Camera(camera_index)
        
        # Register config update callback
        self.config_manager.register_callback(self.on_config_update)
        
        # Initialization flags
        self.running = False
        self.face_count = 0
        
        # Initialize detectors
        self.object_detector = YOLODetector("model_ncnn_model")
        self.motion_detector = MotionDetector()
        self.face_detector = FaceDetector(min_detection_confidence=0.8)
        
        # Initialize face tracking
        self.tracked_faces = []
        self.FACE_DUPLICATE_THRESHOLD = 90
        
        # Timing variables
        self.last_detection_time = 0
        self.last_face_save_time = 0
        self.DETECTION_INTERVAL = self.config["system"]["detection_interval"]
        self.FACE_SAVE_INTERVAL = self.config["system"]["face_save_interval"]

        # Alarm tracking
        self.fire_persistence_count = 0
        self.smoke_persistence_count = 0
        self.ALARM_THRESHOLD = self.config["system"]["alarm_threshold"]
        self.alarm_triggered = False
        self.pre_alarm_logged = False
        
        # Sound alarm control
        self.alarm_playing = False
        self.alarm_thread = None
        
        # Replace pygame.mixer.init() with AudioPlayer
        self.alarm = AudioPlayer("alarm.wav")  # Use your compressed sound file
        
        # Initialize Telegram service
        self.telegram_service = TelegramService(self.config_manager)
        
        # System status
        self.system_status = "Normal"
        self.last_detections = {
            "fire": None,
            "smoke": None,
            "motion": None,
            "face": None
        }
        self.recent_faces = []
        self.max_recent_faces = 10
        
        # Detection tracking
        self.no_detection_count = 0
        self.NO_DETECTION_THRESHOLD = 1

        # Store the latest frame for sharing
        self.latest_frame = None
        self.frame_lock = threading.Lock()  # Add a lock for thread-safe access

        # Add statistics tracking
        self.start_time = time.time()
        self.frames_processed = 0

        # Initialize MQ2 sensor
        self.mq2_sensor = MQ2Sensor(pin=17)
        
        # Add MQ2 detection tracking
        self.mq2_gas_detected = False

        # Sliding window for fire/smoke detection
        self.fire_detection_window = []
        self.smoke_detection_window = []
        self.detection_window_size = 10  # Number of frames to track

    def init_telegram_service(self):
        """Reinitialize the Telegram service using the current configuration."""
        self.telegram_service = TelegramService(self.config_manager)
        return self.telegram_service
        
    def cleanup_old_images(self):
        try:
            # Remove all files in faces and detections directories
            for folder in ["faces", "detections"]:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            
            print(f"{Colors.GREEN}Discarded old face and detection images{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error discarding old images: {str(e)}{Colors.RESET}")
            
    def on_config_update(self, new_config):
        """Handle configuration updates"""
        self.config = new_config
        self.DETECTION_INTERVAL = self.config["system"]["detection_interval"]
        self.FACE_SAVE_INTERVAL = self.config["system"]["face_save_interval"]
        self.ALARM_THRESHOLD = self.config["system"]["alarm_threshold"]
        
        # Explicitly reload Telegram service config and reinitialize the bot
        self.telegram_service.reload_config()
        
        print(f"{Colors.CYAN}Configuration updated. Telegram status: {self.telegram_service.is_enabled()}{Colors.RESET}")
    
    def process_object_detection(self, frame_resized, frame_original):
        
        detected_objects = self.object_detector.detect(frame_resized)
        fire_detected = False
        smoke_detected = False

        if detected_objects and len(detected_objects) > 0:
            classes = detected_objects.cls.tolist()
            confidences = detected_objects.conf.tolist()
            print(f"{Colors.GREEN}[{datetime.now().strftime('%H:%M:%S')}] YOLO detected {len(classes)} object(s):{Colors.RESET}")

            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                label = ""
                box_color = (0, 255, 0)
                if cls == 0 and self.config["detection"]["fire"]:
                    label = f"FIRE"
                    box_color = (0, 0, 255)  # Red for fire
                    fire_detected = True
                elif cls == 1 and self.config["detection"]["smoke"]:
                    label = f"SMOKE"
                    box_color = (255, 0, 0)  # Blue for smoke
                    smoke_detected = True

                box = detected_objects[i]
                # Scale bounding box from resized to original frame
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                scale_x = frame_original.shape[1] / frame_resized.shape[1]
                scale_y = frame_original.shape[0] / frame_resized.shape[0]
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                # Make the bounding box cooler with thicker lines and outline
                line_thickness = 1
                outline_color = (255, 255, 255)  # White outline

                # Draw outline
                cv2.rectangle(frame_original, (x1, y1), (x2, y2), outline_color, line_thickness + 2)
                # Draw main box
                cv2.rectangle(frame_original, (x1, y1), (x2, y2), box_color, line_thickness)

                # Add text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                text_color = (255, 255, 255)  # White text
                text_thickness = 2
                
                # Get size of label to create a background
                (text_width, text_height) = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                
                # Make sure the text background is within the image
                text_x = x1
                text_y = y1 - text_height - 5
                if text_y < 0:
                    text_y = y1 + text_height + 5
                
                # Draw a filled rectangle behind the text
                cv2.rectangle(frame_original, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), box_color, -1)

                # Put the text onto the frame
                cv2.putText(frame_original, label, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

                # Save the full frame with bounding box
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detection_filename = f"detections/{label.lower().replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(detection_filename, frame_original)
                print(f"  {Colors.CYAN}Saved detection image: {detection_filename}{Colors.RESET}")

                self.update_alarm_state(fire_detected, smoke_detected, detection_filename)
        else:
            self.update_alarm_state(False, False)
        return fire_detected, smoke_detected

    def run(self):
        self.running = True
        print(f"{Colors.BOLD}{Colors.CYAN}Enhanced detection system started. Press Ctrl+C to exit.{Colors.RESET}")
        
        # Use consistent processing resolution of 320x320
        process_width = 224
        process_height = 224
        
        try:
            while self.running:
                start_time = time.time()
                success, frame = self.camera.read_frame()
                if not success:
                    print(f"{Colors.RED}Error reading frame from camera{Colors.RESET}")
                    time.sleep(0.5)
                    continue
                
                # Increment frames processed
                self.frames_processed += 1

                # Store the latest frame for sharing
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                current_time = time.time()
                
                # Process every frame but use a simplified processing pipeline when under high load
                process_full = time.time() - self.last_detection_time >= self.DETECTION_INTERVAL
                
                if process_full:
                    self.last_detection_time = time.time()
                    
                    # Resize frame once for all processing
                    frame_resized = cv2.resize(frame, (process_width, process_height))
                    
                    # Convert resized frame to RGB for MediaPipe (face detection)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    
                    # Process with motion detection
                    motion_detected = False
                    if self.config_manager.is_detection_enabled("motion"):
                        motion_detected = self.process_motion_detection(frame_resized)
                    
                    # Process with face detection
                    face_detected = False
                    if self.config_manager.is_detection_enabled("face"):
                        face_detected = self.process_face_detection(frame, frame_rgb)
                    
                    # Process with object detection (fire/smoke)
                    fire_detected = False
                    smoke_detected = False
                    if (self.config_manager.is_detection_enabled("fire") or 
                        self.config_manager.is_detection_enabled("smoke")):
                        fire_detected, smoke_detected = self.process_object_detection(frame_resized, frame)
                    
                    # Process MQ2 gas detection
                    self.mq2_gas_detected = self.mq2_sensor.read_sensor()
                    if self.mq2_gas_detected:
                        print(f"{Colors.RED}[{datetime.now().strftime('%H:%M:%S')}] Gas detected by MQ2 sensor!{Colors.RESET}")
                        self.update_alarm_state(fire_detected, smoke_detected, confirmed_by_mq2=True)
                        self.telegram_service.send_gas_alert()
                    else:
                        print(f"{Colors.GREEN}[{datetime.now().strftime('%H:%M:%S')}] No gas detected by MQ2 sensor.{Colors.RESET}")
                    
                    # Reset to normal if no detections
                    if not (fire_detected or smoke_detected or motion_detected or face_detected or self.mq2_gas_detected):
                        self.no_detection_count += 1
                        if (self.no_detection_count > self.NO_DETECTION_THRESHOLD and 
                            self.system_status != "Normal" and 
                            not self.alarm_playing):
                            self.system_status = "Normal"
                            print(f"{Colors.GREEN}No detections, status reset to Normal{Colors.RESET}")
                    else:
                        self.no_detection_count = 0
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting program...{Colors.RESET}")
        finally:
            self.shutdown()
    
    def process_motion_detection(self, frame):
        """Process frame with motion detector without saving motion images"""
        if not self.config["detection"]["motion"]:
            return False

        motion_detected = self.motion_detector.detect(frame)

        if motion_detected:
            print(f"{Colors.BLUE}[{datetime.now().strftime('%H:%M:%S')}] Motion detected{Colors.RESET}")
            # Update status without saving image or sending a Telegram alert
            self.system_status = "Motion Detected"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_detections["motion"] = timestamp
            
            self.telegram_service.send_motion_alert(None)
            
            return True
        return False
    
    def process_face_detection(self, original_frame, frame_rgb):
        """Process frame with MediaPipe face detector"""
        if not self.config["detection"]["face"]:
            return False
            
        face_detections = self.face_detector.detect(frame_rgb)
        current_faces = []
        face_detected = False
        
        if face_detections:
            print(f"{Colors.MAGENTA}[{datetime.now().strftime('%H:%M:%S')}] Found {len(face_detections)} face(s){Colors.RESET}")
            
            for detection_idx, detection in enumerate(face_detections):
                face_roi, face_center, face_coords = self.extract_face_roi(original_frame, detection)
                current_faces.append(face_center)
                
                is_duplicate = False
                for prev_face in self.tracked_faces:
                    if euclidean(prev_face, face_center) < self.FACE_DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break
                
                if not is_duplicate and face_roi is not None and face_roi.size > 0:
                    # Save the cropped face region at the highest resolution
                    x1, y1, x2, y2 = face_coords
                    high_res_face = original_frame[y1:y2, x1:x2]
                    face_filename = f"faces/face_{self.face_count}.jpg"
                    cv2.imwrite(face_filename, high_res_face)
                    self.telegram_service.send_face_alert(face_filename)
                    
                    print(f"  {Colors.MAGENTA}Saved face {detection_idx+1}: {face_filename}{Colors.RESET}")
                    
                    # Update status
                    self.system_status = "Face Detected"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.last_detections["face"] = timestamp
                    
                    # Add to recent faces
                    filename = os.path.basename(face_filename)
                    self.recent_faces.insert(0, filename)
                    if len(self.recent_faces) > self.max_recent_faces:
                        self.recent_faces = self.recent_faces[:self.max_recent_faces]
                    
                    self.face_count += 1
                    self.tracked_faces.append(face_center)
                    face_detected = True
                    
                    # Limit the number of saved face images
                    self.limit_saved_faces()
            
            # Keep only recent faces (memory efficiency)
            self.tracked_faces = current_faces + self.tracked_faces[:10]  # Keep fewer faces for Pi
            
        return face_detected

    def extract_face_roi(self, frame, detection, scale_factor=1.5):
        """Extract face ROI with additional context based on landmarks."""
        ih, iw, _ = frame.shape

        # Get the bounding box from landmarks
        x_min = iw
        y_min = ih
        x_max = 0
        y_max = 0

        for landmark in detection.landmark:
            x = int(landmark.x * iw)
            y = int(landmark.y * ih)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Expand the bounding box using the scale factor
        w = x_max - x_min
        h = y_max - y_min
        center_x = x_min + w // 2
        center_y = y_min + h // 2

        crop_w = int(w * scale_factor)
        crop_h = int(h * scale_factor)

        # Ensure the box is taller than wide
        if crop_h < crop_w * 1.2:
            crop_h = int(crop_w * 1.2)

        # Calculate new box coordinates
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(iw, x1 + crop_w)
        y2 = min(ih, y1 + crop_h)

        # Adjust if box hits frame boundaries
        if x1 == 0:
            x2 = min(iw, crop_w)
        if y1 == 0:
            y2 = min(ih, crop_h)
        if x2 == iw:
            x1 = max(0, iw - crop_w)
        if y2 == ih:
            y1 = max(0, ih - crop_h)

        face_center = (center_x, center_y)
        face_roi = frame[y1:y2, x1:x2]

        return face_roi, face_center, (x1, y1, x2, y2)
    
    def limit_saved_faces(self):
        """Limit the number of saved face images to prevent disk space issues"""
        max_saved_faces = self.config["system"]["max_saved_faces"]
        face_files = sorted([f for f in os.listdir("faces") if f.startswith("face_")], 
                           key=lambda x: os.path.getmtime(os.path.join("faces", x)))
        
        # If we have more than the maximum, delete the oldest ones
        if len(face_files) > max_saved_faces:
            files_to_delete = face_files[:(len(face_files) - max_saved_faces)]
            for file in files_to_delete:
                try:
                    os.remove(os.path.join("faces", file))
                    print(f"{Colors.YELLOW}Removed old face image: {file}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error removing old face image: {str(e)}{Colors.RESET}")
    
    def update_alarm_state(self, fire, smoke, image_path=None, confirmed_by_mq2=False):
        """Enhanced alarm state logic with sliding window and persistence tracking."""
        # Update sliding windows
        self.fire_detection_window.append(fire)
        self.smoke_detection_window.append(smoke)

        # Trim windows to the defined size
        if len(self.fire_detection_window) > self.detection_window_size:
            self.fire_detection_window.pop(0)
        if len(self.smoke_detection_window) > self.detection_window_size:
            self.smoke_detection_window.pop(0)

        # Determine if fire/smoke is persistently detected
        fire_persist = any(self.fire_detection_window)
        smoke_persist = any(self.smoke_detection_window)

        # Handle gas detection
        if confirmed_by_mq2:
            print(f"{Colors.BOLD}{Colors.RED}⚠️ GAS DETECTED! Triggering alarm...{Colors.RESET}")
            self.system_status = "Gas Detected"
            self.alarm_triggered = True
            self.play_alarm_sound()
            self.telegram_service.send_gas_alert()
            return  # Exit early to prevent resetting to "Normal"

        # Trigger alarm if fire or smoke is persistently detected
        if (fire_persist or smoke_persist) and not self.alarm_triggered:
            alarm_type = ""
            if fire_persist and smoke_persist:
                alarm_type = "FIRE & SMOKE"
                self.system_status = "Fire & Smoke Detected"
            elif fire_persist:
                alarm_type = "FIRE"
                self.system_status = "Fire Detected"
            else:
                alarm_type = "SMOKE"
                self.system_status = "Smoke Detected"

            print(f"{Colors.BOLD}{Colors.RED}🔥🔥 ALARM STATE: PERSISTENT {alarm_type} DETECTED! 🔥🔥{Colors.RESET}")
            self.alarm_triggered = True
            self.play_alarm_sound()  # Start alarm

            # Send Telegram notification with the image
            if fire_persist:
                print(f"Sending fire alert with image: {image_path}")
                self.telegram_service.send_fire_alert(image_path)
            elif smoke_persist:
                print(f"Sending smoke alert with image: {image_path}")

        # Stop alarm if no fire, smoke, or gas is detected in the sliding window
        elif not fire_persist and not smoke_persist and not confirmed_by_mq2 and self.alarm_triggered:
            print(f"{Colors.BOLD}{Colors.GREEN}✅ Normal State: No persistent Fire, Smoke, or Gas detected. Resetting alarm...{Colors.RESET}")
            self.stop_alarm()  # Stop the alarm immediately
            self.alarm_triggered = False
            self.system_status = "Normal"

    def play_alarm_sound(self):
        """Play alarm sound in a thread-safe manner"""
        if self.alarm_playing:
            return  # Already playing

        self.alarm_playing = True

        def alarm_loop():
            try:
                while self.alarm_playing and self.running:
                    self.alarm.play_sound()
                    time.sleep(5)  # Wait for sound to complete or loop
            except Exception as e:
                print(f"Error in alarm loop: {str(e)}")
            finally:
                self.alarm.stop_sound()
                self.alarm_playing = False

        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1)

        self.alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
        self.alarm_thread.start()

    def stop_alarm(self):
        """Safely stop the alarm immediately"""
        self.alarm_playing = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1)
        self.alarm.stop_sound()  # Ensure the sound stops immediately
        self.system_status = "Normal"
        self.alarm_triggered = False
        self.pre_alarm_logged = False
        self.fire_persistence_count = 0
        self.smoke_persistence_count = 0
        print(f"{Colors.GREEN}Alarm stopped and system reset to Normal state.{Colors.RESET}")

    def shutdown(self):
        """Clean shutdown of the system"""
        self.running = False
        if self.camera:
            self.camera.release()  # Ensure the camera is released
        # Make sure to stop any playing alarm and clean up resources
        self.stop_alarm()
        self.alarm.cleanup()
        
        print(f"{Colors.BOLD}{Colors.CYAN}Session summary:{Colors.RESET}")
        print(f"- Faces detected and saved: {Colors.MAGENTA}{self.face_count}{Colors.RESET}")
        print(f"- Detection images saved: {Colors.GREEN}{len(os.listdir('detections'))}{Colors.RESET}")
        print(f"{Colors.BOLD}Resources released. Program terminated.{Colors.RESET}")
    
    def get_status(self):
        """Get current system status"""
        return {
            "status": self.system_status,
            "last_detections": self.last_detections,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_faces(self):
        """Get list of recent faces"""
        return self.recent_faces
    
    def test_telegram(self):
        """Send a test message to verify Telegram configuration"""
        return self.telegram_service.send_test_message()

    def get_latest_frame(self):
        """Get the latest frame captured by the camera."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()  # Return a copy to avoid external modification
            else:
                return None

    def get_statistics(self):
        """Get system statistics including uptime and frames processed"""
        current_time = time.time()
        uptime_seconds = int(current_time - self.start_time)
        
        # Calculate hours, minutes, seconds
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return {
            "uptime": uptime_formatted,
            "frames_processed": self.frames_processed
        }

# Create a global detection system instance
detection_system = None

# Function to initialize the detection system
def init_detection_system(config_path="config.json"):
    global detection_system
    if detection_system is None:
        detection_system = DetectionSystem(config_path)
    return detection_system

# Function to start the detection system in a separate thread
def start_detection_system():
    global detection_system
    if detection_system is None:
        detection_system = init_detection_system()
    
    # Start in a separate thread
    thread = threading.Thread(target=detection_system.run, daemon=True)
    thread.start()
    return thread

# Main function
if __name__ == "__main__":
    # Create output directories
    os.makedirs("faces", exist_ok=True)
    os.makedirs("detections", exist_ok=True)
    
    # Create and start the detection system
    system = init_detection_system()
    
    print("Starting detection system...")
    system.run()
