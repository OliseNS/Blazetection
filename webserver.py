import os
import threading
import time
import queue
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import pyttsx3
import cv2
import asyncio
from flask_socketio import SocketIO
import base64

# Import the detection system
from detection_system import init_detection_system, start_detection_system

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Initialize Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

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
    """Render the main dashboard page"""
    return render_template('index.html')

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
    """Get a list of detection images"""
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


def stream_frames():
    """Continuously stream frames to WebSocket clients."""
    while True:
        if connected_clients > 0:
            frame = detection_system.get_latest_frame()
            if frame is not None:
                # Resize the frame to a higher resolution for better quality
                frame_resized = cv2.resize(frame, (640, 480))  # Adjust resolution as needed
                
                # Encode the frame as JPEG with higher quality
                _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                # Convert to base64 for WebSocket transmission
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Emit the frame to all connected clients
                socketio.emit('video_frame', {'frame': frame_data})
        
        # Adjust frame rate (e.g., ~60 FPS)
        socketio.sleep(1/60)

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket client connections."""
    global connected_clients
    connected_clients += 1
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket client disconnections."""
    global connected_clients
    connected_clients -= 1
    print("Client disconnected")

def start_web_server(host='0.0.0.0', port=8080):
    """Start the Flask web server"""
    global detection_thread

    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = start_detection_system()
        time.sleep(2)
        tts_queue.put("System started successfully.")
        
        # Send Telegram welcome message after system is ready
        detection_system.telegram_service.send_welcome_message()

    # Start the frame streaming thread
    socketio.start_background_task(stream_frames)

    # Start the Flask-SocketIO server
    socketio.run(app, host=host, port=port)

if __name__ == '__main__':
    os.makedirs('faces', exist_ok=True)
    os.makedirs('detections', exist_ok=True)

    tts_queue.put("Starting the web server and detection system. Please wait.")  # TTS announcement
    print(f"Starting web server on http://0.0.0.0:8080")
    print(f"Detection system will run in the background")
    print(f"Press Ctrl+C to exit")

    start_web_server()
