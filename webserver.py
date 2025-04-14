import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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
tts_engine.setProperty('rate', 130)
tts_engine.setProperty('volume', 0.7)

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
        
        # Use a completely new process for TTS to avoid threading issues
        def speak_text():
            try:
                os.system(f'python -c "import pyttsx3; engine = pyttsx3.init(); engine.say(\'{text}\'); engine.runAndWait()"')
            except Exception as e:
                print(f"TTS subprocess error: {str(e)}")
        
        # Run in a separate thread to not block the API response
        threading.Thread(target=speak_text).start()
        
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

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Returns a multipart/x-mixed-replace response with JPEG frames."""
    def generate():
        frame_interval = 0.02  # Target FPS (adjust as needed)
        last_frame_time = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Control frame rate
            if elapsed < frame_interval:
                # Wait until it's time for the next frame
                time.sleep(max(0, frame_interval - elapsed))
                continue
                
            last_frame_time = current_time
            
            if detection_system is not None:
                frame = detection_system.get_latest_frame()
                if frame is not None:
                    # Reduce size for efficiency - scale down to 50% of original size
                    frame_resized = cv2.resize(frame, (0, 0), fx=1, fy=1)
                    
                    # Encode frame as JPEG with reduced quality for better performance
                    ret, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    # No frame available, add a small delay to avoid CPU spinning
                    time.sleep(0.1)
            else:
                # No detection system, add a small delay
                time.sleep(0.1)

    response = Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Add Cache-Control headers to prevent browser buffering
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response

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
