import threading
import cv2
import os
import time
import json
import shutil
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import euclidean
from datetime import datetime
import pygame
import telegram
import asyncio
import telegram 
import flask
from flask import Flask, render_template, request, jsonify, send_from_directory
import pyaudio
import numpy as np
import struct
import subprocess

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

# Telegram integration
class TelegramService:
    """
    Service for sending notifications and images to Telegram
    """
    def __init__(self, config_manager):
        # Existing initialization
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.bot = None
        self.last_notification_time = {}  # Track last notification time by type
        self.last_global_notification_time = 0  # Global cooldown tracker
        self.setup_bot()
        
    def reload_config(self):
        """Reload configuration and update bot if needed"""
        old_config = self.config
        new_config = self.config_manager.get_config()
        
        # Only set up the bot if the configuration has actually changed
        if (old_config["telegram"]["enabled"] != new_config["telegram"]["enabled"] or
            old_config["telegram"]["token"] != new_config["telegram"]["token"] or
            old_config["telegram"]["chat_id"] != new_config["telegram"]["chat_id"]):
            self.config = new_config
            self.setup_bot()
            print(f"Telegram configuration reloaded. Enabled: {self.is_enabled()}")
        else:
            # Just update cooldown without bot recreation
            self.config = new_config
    
    def setup_bot(self):
        """Initialize Telegram bot with current configuration"""
        if not self.config["telegram"]["enabled"] or not self.config["telegram"]["token"]:
            self.bot = None
            return
            
        try:
            self.bot = telegram.Bot(token=self.config["telegram"]["token"])
            print(f"Telegram bot initialized with token: {self.config['telegram']['token'][:5]}...")
        except Exception as e:
            print(f"Error initializing Telegram bot: {str(e)}")
            self.bot = None
    
    def is_enabled(self):
        """Check if Telegram notifications are enabled"""
        return (self.config["telegram"]["enabled"] and 
                self.bot is not None and 
                self.config["telegram"]["chat_id"])
    
    def can_send_notification(self, notification_type):
        """
        Check if a notification can be sent for this type, using the shared cooldown value
        but tracking last sent time per type.
        """
        if not self.is_enabled():
            print(f"Telegram notifications not enabled")
            return False

        current_time = time.time()
        cooldown = self.config["telegram"]["cooldown"]

        # Only check per-type cooldown, not global
        last_time = self.last_notification_time.get(notification_type, 0)
        if current_time - last_time >= cooldown:
            return True

        print(f"Telegram notification on cooldown for type: {notification_type}. " +
              f"Next notification in {cooldown - (current_time - last_time):.1f} seconds")
        return False
    
    async def send_notification_async(self, message, image_path=None, notification_type="general"):
        """
        Send a notification message and optionally an image to Telegram using asyncio
        """
        if not self.is_enabled() or not self.can_send_notification(notification_type):
            return False
            
        # Update last notification times
        current_time = time.time()
        self.last_notification_time[notification_type] = current_time
        self.last_global_notification_time = current_time
        
        try:
            chat_id = self.config["telegram"]["chat_id"]
            
            # First send the message with image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    await self.bot.send_photo(
                        chat_id=chat_id, 
                        photo=image_file, 
                        caption=message,
                    )
                print(f"Telegram image sent: {image_path}")
            else:
                await self.bot.send_message(
                    chat_id=chat_id, 
                    text=message,
                )
                print(f"Telegram message sent without image")
                    
            print(f"Telegram notification sent: {notification_type}")
            return True
        except Exception as e:
            print(f"Error sending Telegram notification: {str(e)}")
            # Force re-initialization of the bot on error
            self.setup_bot()
            return False
    
    def send_notification(self, message, image_path=None, notification_type="general"):
        """
        Non-async wrapper for send_notification_async that doesn't block
        """
        print(f"Queuing {notification_type} notification via Telegram")
        
        # Make sure we have the latest config, but only if needed
        if not self.is_enabled():
            # First quick check without reloading
            self.reload_config()  # Only reload if current config shows disabled
            if not self.is_enabled():
                print(f"Telegram is not properly configured or enabled")
                return False
        
        # Create and start a thread for sending the notification
        telegram_thread = threading.Thread(
            target=self._send_notification_thread,
            args=(message, image_path, notification_type),
            daemon=True
        )
        telegram_thread.start()
        return True  # Return immediately, notification sent asynchronously
    
    def _send_notification_thread(self, message, image_path, notification_type):
        """Thread function to handle the async notification without blocking"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function in the new loop
            loop.run_until_complete(
                self.send_notification_async(message, image_path, notification_type)
            )
        except Exception as e:
            print(f"Exception in Telegram notification thread: {str(e)}")
        finally:
            # Close the loop
            loop.close()
    
    def send_fire_alert(self, image_path=None):
        """Send fire detection alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚨 FIRE DETECTED! 🔥\nTimestamp: {timestamp}\nImmediate action is required!"
        print(f"Sending fire alert with image: {image_path}")
        return self.send_notification(message, image_path, "fire")
    
    def send_smoke_alert(self, image_path=None):
        """Send smoke detection alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"⚠️ SMOKE DETECTED! 💨\nTimestamp: {timestamp}\nCheck for fire!"
        print(f"Sending smoke alert with image: {image_path}")
        return self.send_notification(message, image_path, "smoke")
    
    def send_face_alert(self, image_path=None):
        """Send face detection alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"👁️ FACE DETECTED! 👁️\nTimestamp: {timestamp}"
        return self.send_notification(message, image_path, "face")
    
    def send_motion_alert(self, image_path=None):
        """Send motion detection alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🔄 MOTION DETECTED! \nTimestamp: {timestamp}"
        return self.send_notification(message, image_path, "motion")
    
    def send_test_message(self):
        """Send a test message to verify Telegram configuration"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"✅ TEST MESSAGE ✅\nYour Telegram integration is working correctly!\nTimestamp: {timestamp}"
        return self.send_notification(message, None, "test")

# Camera handling
class Camera:
    def __init__(self, camera_index=0):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"{Colors.RED}Error accessing webcam{Colors.RESET}")
        
        # Set camera properties to the maximum supported resolution and FPS for the system
        max_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)  # Set to maximum supported width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)  # Set to maximum supported height
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size for low latency
        self.cap.set(cv2.CAP_PROP_FPS, max_fps)  # Set to maximum supported FPS for smooth video
    
    def read_frame(self):
        """Read a frame from the camera"""
        return self.cap.read()
    
    def release(self):
        """Release camera resources"""
        self.cap.release()

# YOLO detector
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.65, iou_threshold=0.55):
        """Initialize YOLO object detector"""
        self.model = YOLO(model_path)
        self.CONF_THRESHOLD = conf_threshold
        self.IOU_THRESHOLD = iou_threshold
    
    def detect(self, frame):
        """Detect objects in frame using YOLO"""
        # Use 320x320 image size for better performance on Pi
        results = self.model.predict(frame, imgsz=224, conf=self.CONF_THRESHOLD, iou=self.IOU_THRESHOLD)
        return results[0].boxes

# Pose detector
class PoseDetector:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        """Initialize MediaPipe pose detector"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # Use lightweight model for Pi
        )
    
    def detect(self, frame_rgb):
        """Detect poses in frame"""
        # Provide image dimensions explicitly
        height, width, _ = frame_rgb.shape
        results = self.pose.process(frame_rgb)
        return results

# Face detector
class FaceDetector:
    def __init__(self, min_detection_confidence=0.97):  # High threshold to reduce false positives
        """Initialize MediaPipe face detector"""
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1  # Use lightweight model for Pi
        )
    
    def detect(self, frame_rgb):
        """Detect faces in frame"""
        results = self.face_detection.process(frame_rgb)
        return results.detections if results.detections else []

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
        
        # Apply background subtraction
        # Convert to grayscale for better performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)
        
        # Apply thresholding to remove noise
        thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                motion_detected = True
                self.last_motion_time = current_time
                break
                
        # Return true if motion was detected recently (within cooldown period)
        return motion_detected or (current_time - self.last_motion_time < self.motion_cooldown)

class SineWaveAlarm:
    def __init__(self):
        self.initialize_audio()

    def initialize_audio(self):
        """Initialize PyAudio instance and check for available devices"""
        try:
            self.p = pyaudio.PyAudio()
            # Check if we have any output devices
            device_count = self.p.get_device_count()
            if device_count == 0:
                raise Exception("No audio devices found")
            
            # Find a valid output device
            self.device_index = None
            for i in range(device_count):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    self.device_index = i
                    break
            
            if self.device_index is None:
                raise Exception("No valid output device found")
                
            self.sample_rate = 44100
            self.duration = 0.45
            self.low_freq = 800
            self.high_freq = 1600
            self.amplitude = 0.5
        except Exception as e:
            print(f"Error initializing audio: {str(e)}")
            self.p = None

    def generate_tone(self, frequency, duration=None):
        if duration is None:
            duration = self.duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t) * self.amplitude
        return (wave * 32767).astype(np.int16).tobytes()

    def fire_alarm_siren(self, duration=0.5):
        try:
            if self.p is None:
                self.initialize_audio()
            if self.p is None:  # If still None after reinitialization
                return

            stream = self.p.open(format=pyaudio.paInt16,
                               channels=1,
                               rate=self.sample_rate,
                               output=True,
                               output_device_index=self.device_index)
            
            tone_low = self.generate_tone(self.low_freq, duration)
            tone_high = self.generate_tone(self.high_freq, duration)
            stream.write(tone_high)
            stream.write(tone_low)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error playing alarm: {str(e)}")
            # Cleanup and reinitialize on error
            self.cleanup()
            self.initialize_audio()

    def cleanup(self):
        """Safely cleanup PyAudio resources"""
        try:
            if hasattr(self, 'p') and self.p is not None:
                self.p.terminate()
                self.p = None
        except Exception as e:
            print(f"Error during audio cleanup: {str(e)}")

# Main detection system
class DetectionSystem:
    def __init__(self, config_path="config.json"):
        # Clean up old face images on startup
        self.cleanup_old_images()
        
        # Create output directories
        os.makedirs("faces", exist_ok=True)
        os.makedirs("detections", exist_ok=True)
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Register config update callback
        self.config_manager.register_callback(self.on_config_update)
        
        # Initialization flags
        self.running = False
        self.face_count = 0
        
        # Initialize camera
        camera_index = self.config["system"]["camera_index"]
        self.camera = Camera(camera_index)
        
        # Initialize detectors
        self.object_detector = YOLODetector("model_ncnn_model")
        self.motion_detector = MotionDetector()
        self.face_detector = FaceDetector()
        
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
        
        # Replace pygame.mixer.init() with SineWaveAlarm
        self.alarm = SineWaveAlarm()
        
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
        self.NO_DETECTION_THRESHOLD = 3  # Number of consecutive no-detection frames to reset to Normal

        # Store the latest frame for sharing
        self.latest_frame = None

        # Add statistics tracking
        self.start_time = time.time()
        self.frames_processed = 0
        
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
    
    def run(self):
        self.running = True
        print(f"{Colors.BOLD}{Colors.CYAN}Enhanced detection system started. Press Ctrl+C to exit.{Colors.RESET}")
        
        # Lower resolution for processing to improve performance
        process_width = 224  # Match the YOLO model's input size for optimal performance
        process_height = 224
        
        # Adaptive timing variables
        processing_times = []
        max_processing_times = 10  # Number of times to average
        
        try:
            while self.running:
                start_time = time.time()
                success, frame = self.camera.read_frame()
                if not success:
                    print(f"{Colors.RED}Error reading frame from camera{Colors.RESET}")
                    time.sleep(0.5)  # Reduced wait time
                    continue
                
                # Increment frames processed
                self.frames_processed += 1

                # Store the latest frame for sharing
                self.latest_frame = frame.copy()

                current_time = time.time()
                
                # Process every frame but use a simplified processing pipeline when under high load
                process_full = current_time - self.last_detection_time >= self.DETECTION_INTERVAL
                
                if process_full:
                    self.last_detection_time = current_time
                    # Full processing pipeline
                    
                    # Convert to RGB for MediaPipe (face detection)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with motion detection
                    motion_detected = False
                    if self.config_manager.is_detection_enabled("motion"):
                        motion_detected = self.process_motion_detection(frame)
                    
                    # Process with face detection
                    face_detected = False
                    if self.config_manager.is_detection_enabled("face"):
                        face_detected = self.process_face_detection(frame, frame_rgb)
                    
                    # Process with object detection (fire/smoke)
                    fire_detected = False
                    smoke_detected = False
                    if (self.config_manager.is_detection_enabled("fire") or 
                        self.config_manager.is_detection_enabled("smoke")):
                        # Resize for better performance with YOLO
                        frame_resized = cv2.resize(frame, (process_width, process_height))
                        fire_detected, smoke_detected = self.process_object_detection(frame_resized, frame)
                    
                    # Reset to normal if no detections
                    if not (fire_detected or smoke_detected or motion_detected or face_detected):
                        self.no_detection_count += 1
                        if (self.no_detection_count > self.NO_DETECTION_THRESHOLD and 
                            self.system_status != "Normal" and 
                            not self.alarm_playing):
                            self.system_status = "Normal"
                            print(f"{Colors.GREEN}No detections, status reset to Normal{Colors.RESET}")
                    else:
                        self.no_detection_count = 0
                
                # Adaptive sleep based on processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                if len(processing_times) > max_processing_times:
                    processing_times.pop(0)
                
                avg_processing_time = sum(processing_times) / len(processing_times)
                sleep_time = max(0.01, self.DETECTION_INTERVAL/2 - avg_processing_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting program...{Colors.RESET}")
        finally:
            self.shutdown()
    
    def process_object_detection(self, frame_resized, frame_original):
        detected_objects = self.object_detector.detect(frame_resized)
        fire_detected = False
        smoke_detected = False

        if detected_objects and len(detected_objects) > 0:
            classes = detected_objects.cls.tolist()
            confidences = detected_objects.conf.tolist()
            print(f"{Colors.GREEN}[{datetime.now().strftime('%H:%M:%S')}] YOLO detected {len(classes)} object(s):{Colors.RESET}")

            # Draw boxes on the original high-res frame
            detection_frame = frame_original.copy()

            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                label = ""
                box_color = (0, 255, 0)
                if cls == 0 and self.config["detection"]["fire"]:
                    label = f"FIRE Detected"
                    box_color = (0, 0, 255)
                    fire_detected = True
                elif cls == 1 and self.config["detection"]["smoke"]:
                    label = f"SMOKE Detected"
                    box_color = (0, 0, 255)
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

                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(detection_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if fire_detected or smoke_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detection_filename = f"detections/detection_{timestamp}.jpg"
                cv2.imwrite(detection_filename, detection_frame)
                print(f"  {Colors.CYAN}Saved detection image: {detection_filename}{Colors.RESET}")
                self.update_alarm_state(fire_detected, smoke_detected, detection_filename)
            else:
                self.update_alarm_state(False, False)
        else:
            self.update_alarm_state(False, False)
        return fire_detected, smoke_detected
    
    def process_motion_detection(self, frame):
        """Process frame with motion detector"""
        if not self.config["detection"]["motion"]:
            return False
        
        motion_detected = self.motion_detector.detect(frame)
        
        if motion_detected:
            print(f"{Colors.BLUE}[{datetime.now().strftime('%H:%M:%S')}] Motion detected{Colors.RESET}")
            
            # Update status
            self.system_status = "Motion Detected"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_detections["motion"] = timestamp
            
            # Save motion detection image
            detection_filename = f"detections/motion_{timestamp}.jpg"
            cv2.imwrite(detection_filename, frame)
            
            # Send Telegram notification if enabled
            self.telegram_service.send_motion_alert(detection_filename)
            
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
                face_roi, face_center = self.extract_face_roi(original_frame, detection)
                current_faces.append(face_center)
                
                is_duplicate = False
                for prev_face in self.tracked_faces:
                    if euclidean(prev_face, face_center) < self.FACE_DUPLICATE_THRESHOLD:
                        is_duplicate = True
                        break
                
                if not is_duplicate and face_roi is not None and face_roi.size > 0:
                    face_filename = f"faces/face_{self.face_count}.jpg"
                    cv2.imwrite(face_filename, face_roi)
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
    
    def extract_face_roi(self, frame, detection, scale_factor=1.5):
        """Extract face ROI with additional context"""
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        
        # Basic face detection box
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        
        # Get the face center 
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Create a much larger box centered on the face
        crop_w = int(w * scale_factor)
        crop_h = int(h * scale_factor)
        
        # Make taller than wide to ensure full head capture
        if crop_h < crop_w * 1.2:
            crop_h = int(crop_w * 1.2)
        
        # Calculate new box coordinates, centered on the face
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(iw, x1 + crop_w)
        y2 = min(ih, y1 + crop_h)
        
        # Adjust if box is hitting frame boundaries
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
        
        return face_roi, face_center
    
    def update_alarm_state(self, fire, smoke, image_path=None):
        """Enhanced alarm state logic with persistence tracking, sound, and notifications"""
        if fire and self.config["detection"]["fire"]:
            self.fire_persistence_count += 1
            self.last_fire_smoke_time = time.time()
            if self.fire_persistence_count == self.ALARM_THRESHOLD - 1 and not self.pre_alarm_logged:
                self.system_status = "FIRE PRE-ALERT"
                print(f"{Colors.YELLOW}Fire pre-alert triggered!{Colors.RESET}")
                self.pre_alarm_logged = True
            elif self.fire_persistence_count >= self.ALARM_THRESHOLD:
                self.system_status = "FIRE DETECTED!"
                self.last_detections["fire"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.telegram_service.send_fire_alert(image_path)
                self.play_alarm_sound()
        else:
            self.fire_persistence_count = max(0, self.fire_persistence_count - 1)

        if smoke and self.config["detection"]["smoke"]:
            self.smoke_persistence_count += 1
            self.last_fire_smoke_time = time.time()
            if self.smoke_persistence_count == self.ALARM_THRESHOLD - 1 and not self.pre_alarm_logged:
                self.system_status = "SMOKE PRE-ALERT"
                print(f"{Colors.YELLOW}Smoke pre-alert triggered!{Colors.RESET}")
                self.pre_alarm_logged = True
            elif self.smoke_persistence_count >= self.ALARM_THRESHOLD:
                self.system_status = "SMOKE DETECTED!"
                self.last_detections["smoke"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.telegram_service.send_smoke_alert(image_path)
                self.play_alarm_sound()
        else:
            self.smoke_persistence_count = max(0, self.smoke_persistence_count - 1)

        # Stop alarm if no fire or smoke has been detected for 3 seconds
        current_time = time.time()
        if (self.alarm_playing and not (fire or smoke) and
                current_time - self.last_fire_smoke_time >= 3.0):
            print(f"{Colors.GREEN}No fire/smoke detected for 3 seconds, stopping alarm{Colors.RESET}")
            self.stop_alarm()

    def play_alarm_sound(self):
        """Play alarm sound in a thread-safe manner with short beep cycles"""
        if self.alarm_playing:
            return  # Already playing

        self.alarm_playing = True

        def alarm_loop():
            try:
                while self.alarm_playing and self.running:
                    self.alarm.fire_alarm_siren(duration=0.5)
                    time.sleep(0.05)  # Small delay between beeps
            except Exception as e:
                print(f"Error in alarm loop: {str(e)}")
            finally:
                self.alarm_playing = False

        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1)

        self.alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
        self.alarm_thread.start()

    def stop_alarm(self):
        """Safely stop the alarm"""
        self.alarm_playing = False
        # Removed self.alarm.cleanup() here to avoid terminating audio resources while still in use
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1)
            
        self.system_status = "Normal"
        self.alarm_triggered = False
        self.pre_alarm_logged = False
        self.fire_persistence_count = 0
        self.smoke_persistence_count = 0

    def shutdown(self):
        """Clean shutdown of the system"""
        self.running = False
        self.camera.release()
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
        """Get the latest frame captured by the camera"""
        return self.latest_frame

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
