import os
import uuid
import time
import queue
import threading
import subprocess
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

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
    voices_dir = os.path.join(os.getcwd(), 'voices')
    model_path = os.path.join(voices_dir, "en_US-amy-low.onnx")
    config_path = os.path.join(voices_dir, "amyconfig.json")

    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print(f"Piper model or config not found in {voices_dir}.")
        return

    while True:
        text = tts_queue.get()
        if text is None:
            break

        try:
            print(f"[TTS] Speaking: {text}")
            # Use Piper to synthesize audio and pipe it directly to aplay/ffplay
            piper_cmd = [
                "piper",
                "--model", model_path,
                "--config", config_path,
                "--output_file", "-",  # output to stdout
                "--sentence_silence", "0.3"  # slight pause between sentences
            ]
            if sys.platform == "win32":
                # Windows: Save to file and play with powershell
                temp_wav = os.path.join(os.getcwd(), "temp_tts.wav")
                subprocess.run(piper_cmd + ["-f", "-"], input=text, text=True, stdout=open(temp_wav, "wb"))
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_wav}').PlaySync()"])
                os.remove(temp_wav)
            else:
                # Linux/macOS: Stream audio to aplay or ffplay
                player = ["aplay", "-q"] if sys.platform.startswith("linux") else ["ffplay", "-nodisp", "-autoexit", "-"]
                piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                player_proc = subprocess.Popen(player, stdin=piper_proc.stdout)
                piper_proc.stdin.write(text.encode('utf-8'))
                piper_proc.stdin.close()
                piper_proc.wait()
                player_proc.wait()
        except Exception as e:
            print(f"[TTS Error] {str(e)}")
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

@app.route('/api/audio/upload', methods=['POST'])
def upload_audio():
    """Save uploaded audio recording"""
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400
            
        # Create a unique ID and filename
        recording_id = str(uuid.uuid4())
        recordings_dir = os.path.join(os.getcwd(), 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        
        filename = f"{recording_id}.wav"
        filepath = os.path.join(recordings_dir, filename)
        
        # Save the file
        audio_file.save(filepath)
        
        return jsonify({
            "success": True, 
            "id": recording_id,
            "message": "Recording saved successfully"
        }), 200
        
    except Exception as e:
        print(f"Error saving audio recording: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/audio/play/<recording_id>', methods=['POST'])
def play_audio(recording_id):
    """Play the audio recording using aplay"""
    try:
        recordings_dir = os.path.join(os.getcwd(), 'recordings')
        filepath = os.path.join(recordings_dir, f"{recording_id}.wav")
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "Recording not found"}), 404
            
        # Play the audio file in a separate thread to avoid blocking
        def play_audio_file():
            if sys.platform == "win32":
                # Windows
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{filepath}').PlaySync()"])
            else:
                # Linux
                subprocess.run(["aplay", "-q", filepath])
                
        threading.Thread(target=play_audio_file, daemon=True).start()
        
        return jsonify({
            "success": True,
            "message": "Playing audio recording"
        }), 200
        
    except Exception as e:
        print(f"Error playing audio recording: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/audio/delete/<recording_id>', methods=['DELETE'])
def delete_audio(recording_id):
    """Delete an audio recording"""
    try:
        recordings_dir = os.path.join(os.getcwd(), 'recordings')
        filepath = os.path.join(recordings_dir, f"{recording_id}.wav")
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "error": "Recording not found"}), 404
            
        # Delete the file
        os.remove(filepath)
        
        return jsonify({
            "success": True,
            "message": "Recording deleted successfully"
        }), 200
        
    except Exception as e:
        print(f"Error deleting audio recording: {str(e)}")
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
    os.makedirs('recordings', exist_ok=True)
    print(f"Starting web server on http://0.0.0.0:8080")
    print(f"Detection system will run in the background")
    print(f"Press Ctrl+C to exit")

    start_web_server()
    app.run(host='0.0.0.0', port=8080)