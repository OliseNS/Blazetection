import os
import threading
import time
import queue
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import subprocess
import cv2
import asyncio
import base64
import sys

# Import the detection system
from detection_system import init_detection_system, start_detection_system, Camera  # Add Camera here

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')


# Initialize detection system
detection_system = init_detection_system()

# Create a queue for TTS requests
tts_queue = queue.Queue()

# Function to process TTS requests from the queue
def process_tts_queue():
    # Create voices directory if it doesn't exist
    voices_dir = os.path.join(os.getcwd(), 'voices')
    os.makedirs(voices_dir, exist_ok=True)
    
    # Check if Piper and voice models are available
    piper_available = False
    try:
        result = subprocess.run(["piper", "--help"], capture_output=True, text=True)
        piper_available = result.returncode == 0
    except:
        print("Piper TTS not found. Make sure it's installed: pip install piper-tts")
        print("Download voice models from: https://github.com/rhasspy/piper/releases")
    
    # Default voice model path
    model_path = os.path.join(voices_dir, "en_GB-cori-medium.onnx")
    config_path = os.path.join(voices_dir, "coriconfig.json")
    
    # Check if model files exist
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print(f"Piper voice models not found at {voices_dir}")
        print("Download them from: https://github.com/rhasspy/piper/releases")
        piper_available = False
    
    while True:
        text = tts_queue.get()
        if text is None:
            break
        
        try:
            if piper_available:
                # Create a temporary file for the text
                temp_txt = os.path.join(os.getcwd(), "temp_tts.txt")
                temp_wav = os.path.join(os.getcwd(), "temp_tts.wav")
                
                with open(temp_txt, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Run Piper to generate speech
                subprocess.run([
                    "piper",
                    "--model", model_path,
                    "--config", config_path,
                    "--output_file", temp_wav,
                    "-f", temp_txt
                ])
                
                # Play the audio using a system command
                if sys.platform == "win32":
                    subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_wav}').PlaySync()"])
                else:
                    # For Linux, use aplay
                    subprocess.run(["aplay", temp_wav])
                
                # Clean up temporary files
                if os.path.exists(temp_txt):
                    os.remove(temp_txt)
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            else:
                print(f"Piper TTS not available. Skipping speech for: {text}")
                
        except Exception as e:
            print(f"TTS Processing Error: {str(e)}")
            
        finally:
            tts_queue.task_done()

# Start a background thread to process the TTS queue
tts_thread = threading.Thread(target=process_tts_queue, daemon=True)
tts_thread.start()

# Start detection system in a separate thread
detection_thread = None

connected_clients = 0

@app.route('/')
def index():
    """Render the main dashboard page with camera URL from config"""
    config = detection_system.config_manager.get_config()
    camera_url = config["system"].get("camera_url", "")
    # Convert to string if it's a camera index (integer)
    if isinstance(camera_url, int):
        camera_url = str(camera_url)
    elif camera_url is None:
        camera_url = "0"
    return render_template('index.html', camera_url=camera_url)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify(detection_system.config_manager.get_config())

@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        data = request.json
        section = data.get('section')
        values = data.get('values')
        detection_system.config_manager.update_section(section, values)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status = detection_system.system_status
        return jsonify({"status": status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/faces', methods=['GET'])
def get_faces():
    try:
        faces = detection_system.get_faces()
        return jsonify({"faces": faces}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm():
    """Stop the alarm"""
    try:
        detection_system.stop_alarm()
        detection_system.system_status = "Normal"
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/telegram/test', methods=['POST'])
def test_telegram():
    """Send a test message to verify Telegram configuration"""
    try:
        success = detection_system.test_telegram()
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using pyttsx3"""
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"success": False, "error": "No text provided"})

        tts_queue.put(text)

        return jsonify({"success": True, "message": "Text added to TTS queue"})
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        stats = detection_system.get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    try:
        detection_folder = os.path.join(os.getcwd(), 'detections')
        if not os.path.exists(detection_folder):
            os.makedirs(detection_folder, exist_ok=True)
            return jsonify({"detections": []}), 200

        detection_images = sorted(
            [img for img in os.listdir(detection_folder) if img.endswith(('.jpg', '.png'))],
            key=lambda x: os.path.getmtime(os.path.join(detection_folder, x)),
            reverse=True
        )

        return jsonify({"detections": detection_images}), 200
    except Exception as e:
        print(f"Error in get_detections: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/detections/<path:filename>')
def serve_detection(filename):
    """Serve detection images"""
    return send_from_directory('detections', filename)

@app.route('/faces/<path:filename>')
def serve_face(filename):
    """Serve face images"""
    return send_from_directory('faces', filename)

@app.route('/api/restart', methods=['POST'])
def restart_system():
    """Restart the entire application (web server and detection system)"""
    try:
        # Announce restart via TTS
        tts_queue.put("System is restarting. Please wait.")
        
        # Stop the detection system gracefully if it's running
        global detection_thread
        if detection_thread and detection_thread.is_alive():
            detection_system.running = False
            detection_thread.join(timeout=5)  # Wait for up to 2 seconds
        
        # Clean up resources
        if detection_system:
            detection_system.shutdown()
        
        # First send a response that the restart is in progress
        response = jsonify({"success": True, "message": "System is restarting..."})
        
        # Schedule the restart after the response is sent
        def restart_after_response():
            time.sleep(1)  # Give time for the response to be sent
            # Start a new process with the same command and arguments
            subprocess.Popen([sys.executable] + sys.argv)
            # Exit the current process
            os._exit(0)
        
        threading.Thread(target=restart_after_response, daemon=True).start()
        
        return response
        
    except Exception as e:
        print(f"Restart Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/update-camera', methods=['POST'])
def update_camera_url():
    """Update the camera URL in the configuration"""
    try:
        data = request.json
        camera_url = data.get('camera_url')
        
        if not camera_url:
            return jsonify({"success": False, "error": "No camera URL provided"}), 400
        
        # Try to parse as a number (camera index)
        try:
            if camera_url.isdigit():
                camera_url = int(camera_url)
        except:
            pass  # Keep as string if not a valid integer
        
        # Update configuration
        config = detection_system.config_manager.get_config()
        config["system"]["camera_url"] = camera_url
        detection_system.config_manager.update_config(config)
        
        # Restart the camera with the new URL
        if hasattr(detection_system, 'camera') and detection_system.camera:
            detection_system.camera.release()
        detection_system.camera = Camera(camera_url)
        
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error updating camera URL: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def start_web_server(host='0.0.0.0', port=8080):
    """Start the Flask web server"""
    global detection_thread

    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = start_detection_system()
        time.sleep(2)
        tts_queue.put("System started successfully.")
        
        # Send Telegram welcome message after system is ready
        detection_system.telegram_service.send_welcome_message()

if __name__ == '__main__':
    os.makedirs('faces', exist_ok=True)
    os.makedirs('detections', exist_ok=True)
    print(f"Starting web server on http://0.0.0.0:8080")
    print(f"Detection system will run in the background")
    print(f"Press Ctrl+C to exit")

    start_web_server()
    app.run(host='0.0.0.0', port=8080)
