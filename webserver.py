import os
import threading
import time
import queue
from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import sys
import signal
import atexit

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

# Global variables for persistent TTS processes
piper_proc = None
player_proc = None

def preload_tts_model():
    voices_dir = os.path.join(os.getcwd(), 'voices')
    # Use a faster model - consider switching to "en_US-amy-medium" if it exists
    model_path = os.path.join(voices_dir, "en_US-amy-low.onnx")
    config_path = os.path.join(voices_dir, "amyconfig.json")

    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        raise FileNotFoundError(f"Piper model or config not found in {voices_dir}.")

    return model_path, config_path

# Initialize the persistent TTS process
def initialize_tts_process():
    global piper_proc, player_proc
    
    if not preloaded_model_path or not preloaded_config_path:
        print("[TTS] Model or configuration not available. Cannot initialize TTS process.")
        return False
        
    try:
        # Optimize parameters for speed
        piper_cmd = [
            "piper",
            "--model", preloaded_model_path,
            "--config", preloaded_config_path,
            "--output_file", "-",
            "--sentence_silence", "0.05",  # Minimal silence between sentences
            "--length-scale", "0.85",      # Even faster speech rate
            "--stdin-format", "lines"      # Process line by line for lower latency
        ]
        
        # Start the persistent processes
        piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        player_proc = subprocess.Popen(["aplay", "-q"], stdin=piper_proc.stdout)
        
        print("[TTS] Persistent TTS process initialized successfully.")
        return True
    except Exception as e:
        print(f"[TTS Error] Failed to initialize persistent process: {str(e)}")
        return False

# Clean up TTS processes
def cleanup_tts_processes():
    global piper_proc, player_proc
    
    try:
        if piper_proc:
            print("[TTS] Shutting down persistent TTS process...")
            piper_proc.stdin.close()
            piper_proc.terminate()
            piper_proc.wait(timeout=1)
            
        if player_proc:
            player_proc.terminate()
            player_proc.wait(timeout=1)
    except Exception as e:
        print(f"[TTS Error] Error during cleanup: {str(e)}")

# Register cleanup function
atexit.register(cleanup_tts_processes)

# load the TTS model and configuration
try:
    preloaded_model_path, preloaded_config_path = preload_tts_model()
    print("[TTS] Model and configuration preloaded successfully.")
    # Initialize the persistent TTS process
    tts_initialized = initialize_tts_process()
except FileNotFoundError as e:
    print(f"[TTS Error] {str(e)}")
    preloaded_model_path, preloaded_config_path = None, None
    tts_initialized = False

# Function to process TTS requests from the queue
def process_tts_queue():
    global piper_proc, player_proc
    
    if not tts_initialized:
        print("[TTS] TTS not initialized. Skipping TTS processing.")
        return

    while True:
        text = tts_queue.get()
        if text is None:
            break

        try:
            print(f"[TTS] Speaking: {text}")
            
            # Check if process is still alive
            if piper_proc.poll() is not None or player_proc.poll() is not None:
                print("[TTS] Restarting TTS process...")
                cleanup_tts_processes()
                initialize_tts_process()
            
            # Send text to the persistent process
            piper_proc.stdin.write((text + '\n').encode('utf-8'))
            piper_proc.stdin.flush()  # Important to ensure text is processed immediately
            
        except Exception as e:
            print(f"[TTS Error] {str(e)}")
            # Try to reinitialize the process if there's an error
            cleanup_tts_processes()
            initialize_tts_process()
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
            detection_thread.join(timeout=2)  # Wait for up to 2 seconds
        
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
