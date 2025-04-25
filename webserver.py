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
import hashlib

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

# Cache for recently spoken phrases
tts_cache = {}
tts_lock = threading.Lock()

# Function to process TTS requests from the queue
def process_tts_queue():
    voices_dir = os.path.join(os.getcwd(), 'voices')
    model_path = os.path.join(voices_dir, "en_US-amy-low.onnx")
    config_path = os.path.join(voices_dir, "amyconfig.json")

    # Cache directory for synthesized speech
    cache_dir = os.path.join(os.getcwd(), 'tts_cache')
    os.makedirs(cache_dir, exist_ok=True)

    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print(f"Piper model or config not found in {voices_dir}.")
        return

    # Launch a persistent piper process
    piper_cmd = [
        "piper",
        "--model", model_path,
        "--config", config_path,
        "--output_raw"  # Output raw audio data
    ]
    
    # Flag to track if we're busy speaking
    is_speaking = False
    
    while True:
        try:
            # Get the next text to speak
            text = tts_queue.get(block=True)
            if text is None:
                break
                
            # Skip if we're still speaking and this isn't a high-priority message
            if is_speaking and not text.startswith("ALERT:") and not text.startswith("EMERGENCY:"):
                tts_queue.task_done()
                continue
                
            # Create a hash of the text for caching
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{text_hash}.raw")
            
            # Check if we have this text cached
            if os.path.exists(cache_file):
                # Use cached audio
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
            else:
                # Generate new audio
                try:
                    piper_process = subprocess.Popen(
                        piper_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Send text to piper
                    audio_data, stderr = piper_process.communicate(text.encode(), timeout=5)
                    
                    # If successful, cache the result
                    if piper_process.returncode == 0 and audio_data:
                        with open(cache_file, 'wb') as f:
                            f.write(audio_data)
                    else:
                        print(f"[TTS Error] {stderr.decode()}")
                        tts_queue.task_done()
                        continue
                except Exception as e:
                    print(f"[TTS Error] {str(e)}")
                    tts_queue.task_done()
                    continue
            
            # Play the audio with aplay
            is_speaking = True
            try:
                # -q: quiet, -f: format, -r: rate, -c: channels
                aplay_cmd = ["aplay", "-q", "-f", "S16_LE", "-r", "22050", "-c", "1", "-"]
                play_process = subprocess.Popen(
                    aplay_cmd,
                    stdin=subprocess.PIPE
                )
                
                # Write audio data to aplay
                play_process.stdin.write(audio_data)
                play_process.stdin.close()
                
                # Don't wait for it to finish if we have more urgent messages
                if not tts_queue.empty():
                    # If there are more messages, don't wait for this one to finish
                    threading.Thread(target=play_process.wait, daemon=True).start()
                else:
                    # Otherwise wait for the audio to finish playing
                    play_process.wait()
            except Exception as e:
                print(f"[Audio Playback Error] {str(e)}")
            finally:
                is_speaking = False
                
        except Exception as e:
            print(f"[TTS Queue Error] {str(e)}")
        finally:
            tts_queue.task_done()

        # Small delay to prevent CPU thrashing
        time.sleep(0.01)

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
    """Convert text to speech using piper"""
    try:
        data = request.json
        text = data.get('text', '')
        is_emergency = data.get('emergency', False)

        if not text:
            return jsonify({"success": False, "error": "No text provided"})
            
        # Prefix emergency messages for priority handling
        if is_emergency:
            text = f"EMERGENCY: {text}"

        # Add to the queue
        tts_queue.put(text)

        return jsonify({"success": True, "message": "Text added to TTS queue"})
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts/clear', methods=['POST'])
def clear_tts_queue():
    """Clear the TTS queue in case of emergency"""
    try:
        # Empty the queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
                tts_queue.task_done()
            except:
                pass
                
        # Stop any currently playing audio
        subprocess.run(["pkill", "-f", "aplay"])
        
        return jsonify({"success": True, "message": "TTS queue cleared"})
    except Exception as e:
        print(f"TTS Queue Clear Error: {str(e)}")
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
    # Clean up cache to ensure we're not using too much storage
    cache_dir = os.path.join(os.getcwd(), 'tts_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Limit cache to 100 most recent files
    if os.path.exists(cache_dir):
        files = sorted(
            [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.raw')],
            key=os.path.getmtime,
            reverse=True
        )
        
        # Keep only 100 most recent files
        for old_file in files[100:]:
            try:
                os.remove(old_file)
            except:
                pass
    
    os.makedirs('faces', exist_ok=True)
    os.makedirs('detections', exist_ok=True)
    print(f"Starting web server on http://0.0.0.0:8080")
    print(f"Detection system will run in the background")
    print(f"Press Ctrl+C to exit")

    start_web_server()
    app.run(host='0.0.0.0', port=8080)