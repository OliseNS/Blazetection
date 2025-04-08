from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import time
from datetime import datetime
import pyttsx3

# Import the detection system
from detection_system import init_detection_system, start_detection_system

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize detection system
detection_system = init_detection_system()

# Initialize pyttsx3 engine with custom settings
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Start detection system in a separate thread
detection_thread = None

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
    """Update configuration"""
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
    """Get current system status"""
    return jsonify(detection_system.get_status())

@app.route('/api/faces', methods=['GET'])
def get_faces():
    """Get list of recent faces"""
    return jsonify({
        "faces": detection_system.get_faces()
    })

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm():
    """Stop the alarm"""
    try:
        detection_system.stop_alarm()
        # Reset status to Normal when alarm is manually stopped
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
        
        # Create a new engine instance for each request
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.setProperty('volume', 0.9)
        
        try:
            # Speak the text with the local engine
            local_engine.say(text)
            local_engine.runAndWait()
        finally:
            # Make sure to properly clean up the engine
            try:
                local_engine.endLoop()
            except:
                pass
            local_engine.stop()
            del local_engine
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"TTS Error: {str(e)}")  # Log the error
        return jsonify({"success": False, "error": str(e)})

@app.route('/faces/<path:filename>')
def serve_face(filename):
    """Serve face images"""
    return send_from_directory('faces', filename)

@app.route('/detections/<path:filename>')
def serve_detection(filename):
    """Serve detection images"""
    return send_from_directory('detections', filename)

def start_web_server(host='0.0.0.0', port=8080):
    """Start the Flask web server"""
    global detection_thread
    
    # Start detection system if not already running
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = start_detection_system()
        # Give the detection system time to initialize
        time.sleep(1)
    
    # Start Flask server
    app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
    # Create required directories
    os.makedirs('faces', exist_ok=True)
    os.makedirs('detections', exist_ok=True)
    
    print(f"Starting web server on http://0.0.0.0:8080")
    print(f"Detection system will run in the background")
    print(f"Press Ctrl+C to exit")
    
    # Start web server
    start_web_server()
