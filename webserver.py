import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import pyttsx3

# Import the detection system functions
from detection_system import init_detection_system, start_detection_system

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize detection system
detection_system = init_detection_system()

# Initialize pyttsx3 engine with custom settings
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 130)
tts_engine.setProperty('volume', 0.7)

# Global variables for detection and streaming
detection_thread = None

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config = detection_system.config_manager.get_config()
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration."""
    try:
        data = request.json
        section = data.get('section')
        values = data.get('values')
        if section and values:
            success = detection_system.config_manager.update_section(section, values)
            return jsonify({"success": success})
        else:
            return jsonify({"success": False, "error": "Missing section or values"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status."""
    return jsonify(detection_system.get_status())

@app.route('/api/faces', methods=['GET'])
def get_faces():
    """Get list of recent faces."""
    faces = detection_system.get_faces()
    return jsonify({"faces": faces})

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm():
    """Stop the alarm and reset the system status to Normal."""
    try:
        detection_system.stop_alarm()
        detection_system.system_status = "Normal"
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/telegram/test', methods=['POST'])
def test_telegram():
    """Send a test message verifying Telegram configuration."""
    try:
        success = detection_system.test_telegram()
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using pyttsx3."""
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({"success": False, "error": "No text provided"})

        def speak_text():
            try:
                # Using a new process for TTS to avoid threading issues
                os.system(f'python -c "import pyttsx3; engine = pyttsx3.init(); engine.say(\'{text}\'); engine.runAndWait()"')
            except Exception as e:
                print(f"TTS subprocess error: {str(e)}")

        threading.Thread(target=speak_text).start()
        return jsonify({"success": True})
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics such as uptime and frames processed."""
    if detection_system is not None:
        stats = detection_system.get_statistics()
        stats.update({"status": detection_system.system_status})
        return jsonify(stats)
    else:
        return jsonify({"error": "Detection system not initialized"})

@app.route('/faces/<path:filename>')
def serve_face(filename):
    """Serve saved face images."""
    return send_from_directory('faces', filename)

@app.route('/detections/<path:filename>')
def serve_detection(filename):
    """Serve saved detection images."""
    return send_from_directory('detections', filename)

@app.route('/video_feed')
def video_feed():
    """Provide a video streaming route."""
    def generate():
        while True:
            frame = detection_system.get_latest_frame()
            if frame is None:
                time.sleep(0.05)  # small delay to prevent tight looping
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_web_server(host='0.0.0.0', port=8080):
    """Start the detection system and Flask web server."""
    global detection_thread
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = start_detection_system()
        # Give the detection system time to initialize
        time.sleep(1)
    print("Starting web server on http://0.0.0.0:8080")
    print("Detection system will run in the background")
    print("Press Ctrl+C to exit")
    app.run(host=host, port=port, threaded=True, processes=1, debug=True)

if __name__ == '__main__':
    os.makedirs('faces', exist_ok=True)
    os.makedirs('detections', exist_ok=True)
    start_web_server()
