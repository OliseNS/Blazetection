import os
import threading
import time
import queue
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import pyttsx3
import cv2
import asyncio
import base64

# Import the detection system
from detection_system import init_detection_system, start_detection_system

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
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            # Initialize a new pyttsx3 engine instance
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)  # Adjust speech rate as needed
            tts_engine.setProperty('volume', 1.0)  # Adjust volume as needed
            tts_engine.say(text)
            tts_engine.runAndWait()
            tts_engine.stop()
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
    camera_url = detection_system.config_manager.get_config()["system"].get("camera_url", "http://192.168.1.225:5000")
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

@app.route('/api/telegram/reload', methods=['POST'])
def reload_telegram():
    """Reload Telegram configuration without restarting the system."""
    try:
        # Update the telegram configuration
        detection_system.telegram_service.reload_config()
        # Reinitialize the telegram service with the new configuration
        detection_system.telegram_service = detection_system.init_telegram_service()
        # Optionally, send a welcome message to verify the connection
        detection_system.telegram_service.send_welcome_message()
        return jsonify({"success": True, "message": "Telegram configuration reloaded and service reinitialized."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
def restart_detection_system():
    """Restart the detection system by shutting it down and reinitializing it."""
    global detection_thread, detection_system

    try:
        # Gracefully shutdown the current detection system if running
        if detection_system:
            detection_system.shutdown()
        
        # Reinitialize detection system (which also reinitializes the camera)
        from detection_system import init_detection_system, start_detection_system  # Ensure fresh instance
        detection_system = init_detection_system()
        detection_thread = start_detection_system()
        
        # Allow some time for initialization
        time.sleep(2)
        tts_queue.put("Detection system restarted successfully.")
        return jsonify({"success": True, "message": "Detection system restarted successfully."}), 200

    except Exception as e:
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
