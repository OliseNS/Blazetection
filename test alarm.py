import numpy as np
import pyaudio
import time

SAMPLE_RATE = 45000  # Standard sample rate
DURATION = 0.45       # Tone duration per step

p = pyaudio.PyAudio()

def generate_tone(frequency, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * 0.5
    return (wave * 32767).astype(np.int16).tobytes()

def fire_alarm_siren(duration=10):
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)

    low_freq = 800
    high_freq = 1600
    tone_low = generate_tone(low_freq, DURATION)
    tone_high = generate_tone(high_freq, DURATION)

    start_time = time.time()
    while time.time() - start_time < duration:
        stream.write(tone_high)
        stream.write(tone_low)
        

    stream.stop_stream()
    stream.close()
    p.terminate()

try:
    print("🚨 Fire detected! Playing alarm...")
    fire_alarm_siren()
except KeyboardInterrupt:
    p.terminate()
