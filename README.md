# Blazetection

Blazetection is an intelligent real-time detection and alert system, primarily designed to run on devices like the Raspberry Pi. It combines advanced AI-based video analysis with hardware sensors to detect fire, smoke, motion, and faces, then delivers instant notifications and alarms via Telegram and on-device actions.

---

## Features

- **Fire, Smoke, Motion & Face Detection**: Utilizes YOLO, Mediapipe, and custom logic to identify safety risks and people in camera feeds.
- **Real-Time Video Dashboard**: Web-based live-feed interface with processed frame stats, recent detections, and system status.
- **Text-to-Speech (TTS)**: Input text on the dashboard to play audio through the device's speakers, with TTS history and replay.
- **Telegram Integration**: Sends instant alerts, images, and allows test messages via Telegram bots.
- **Alarm System**: Triggers a local alarm sound when critical detections occur, with remote stop functionality.
- **MQ2 Gas Sensor Support**: Monitors smoke/gas levels and contributes to alarm triggers.
- **Configurable**: Easily enable/disable detection modules and adjust Telegram settings via the web UI.

---

## Installation

> **Requirements:**  
> - Raspberry Pi (or similar device, recommended for GPIO/MQ2 support)  
> - Python 3.8+  
> - Node.js (for static assets, optional)  
> - Camera compatible with OpenCV  
> - MQ2 gas sensor (optional)  
> - Telegram bot token and chat ID (for notifications)

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/OliseNS/Blazetection.git
    cd Blazetection
    ```

2. **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Install hardware libraries:**
    ```bash
    # GPIO support for Raspberry Pi
    sudo apt-get install python3-gpiozero
    ```

4. **Configure Your Settings:**
    - Edit `config.json` to set detection intervals, alarm thresholds, Telegram credentials, etc.
    - Or use the web dashboard to adjust Telegram and detection settings.

5. **Run the System:**
    ```bash
    python detection_system.py
    ```
    The dashboard will be available at `http://<your-device-ip>:<port>/`.

---

## Usage

- **Dashboard:**  
  Access the web interface to view:
  - Live camera feed.
  - Recent faces & detections.
  - System statistics (uptime, frames processed, status).
  - TTS input & history.
  - Enable/disable detection modules (fire, smoke, motion, face).
  - Configure and test Telegram alerts.
  - Stop the alarm remotely.

- **Telegram Alerts:**  
  Add your bot to a Telegram group or chat and set the `token` and `chat_id` in the dashboard or config file.

- **TTS:**  
  Enter a message in the dashboard for spoken alerts via the device's audio output.

---

## Hardware Integration

- **Camera:**  
  Supports USB, Pi Camera, or any OpenCV-compatible source.

- **MQ2 Sensor:**  
  Wired to GPIO (default: pin 17), used for smoke/gas detection.

---

## Project Structure

- `detection_system.py` &mdash; Main backend for detection logic, alarm management, and integration.
- `static/` &mdash; JavaScript, CSS, and assets for the dashboard.
- `templates/` &mdash; HTML templates for the web UI.
- `config.json` &mdash; System and notification configuration.
- `requirements.txt` &mdash; Python dependencies.

---

## Author

- **OliseNS**  
  [GitHub Profile](https://github.com/OliseNS)

---

## License

This project currently does not specify a license. Please get in touch with the author for usage permissions.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

---

## Acknowledgements

- [YOLO](https://github.com/ultralytics/yolov5) for object detection.
- [Mediapipe](https://google.github.io/mediapipe/) for face detection using face mesh.
- [Telegram Bot API](https://core.telegram.org/bots/api)

---
