import sounddevice as sd
import numpy as np
import time
import threading
import argparse
from typing import Optional

class EmergencyAlarm:
    def __init__(self):
        # Initialize default parameters
        self.high_frequency = 800  # Hz for high tone
        self.low_frequency = 650   # Hz for low tone
        self.sweep_rate = 2.5      # Complete tone cycles per second
        self.volume = 0.8          # 0.0 to 1.0
        self.sample_rate = 44100   # samples per second
        self.harmonics = 3         # Number of harmonics to add for harshness
        
        # State control
        self.is_playing = False
        self.alarm_thread: Optional[threading.Thread] = None
        self.stream = None
    
    def generate_samples(self, num_samples):
        """Generate emergency alarm samples with two-tone pattern"""
        # Create time array for sample generation
        t = np.arange(num_samples) / self.sample_rate
        
        # Create the two-tone pattern (switching between high and low tones)
        # This uses a square wave at sweep_rate to alternate between frequencies
        tone_switch = np.sign(np.sin(2 * np.pi * self.sweep_rate * t))
        
        # Initialize signal array
        signal = np.zeros_like(t, dtype=np.float32)
        
        # For each moment, decide whether to use high or low frequency
        high_mask = tone_switch > 0
        low_mask = ~high_mask
        
        # Generate the base waveform for high tone sections
        for harmonic in range(1, self.harmonics + 1):
            # Using sawtooth-like waveform with harmonics for harshness
            signal[high_mask] += np.sin(2 * np.pi * self.high_frequency * harmonic * t[high_mask]) / harmonic
            
        # Generate the base waveform for low tone sections
        for harmonic in range(1, self.harmonics + 1):
            signal[low_mask] += np.sin(2 * np.pi * self.low_frequency * harmonic * t[low_mask]) / harmonic
        
        # Normalize and apply volume
        signal = signal / (self.harmonics * 0.7) * self.volume
        
        # Add slight distortion for harshness (clipping)
        signal = np.clip(signal * 1.3, -1, 1) * self.volume
        
        # Return as float32 for sounddevice
        return signal.astype(np.float32)
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice stream to continuously provide audio data"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Generate new samples for each callback
        data = self.generate_samples(frames)
        
        # Write data to output buffer
        outdata[:, 0] = data
        
        # If we're no longer playing, stop the stream
        if not self.is_playing:
            raise sd.CallbackStop()
    
    def start_alarm(self):
        """Start the emergency alarm"""
        if self.is_playing:
            print("Alarm is already playing")
            return
        
        self.is_playing = True
        
        # Create and start the audio stream
        self.stream = sd.OutputStream(
            channels=1,  # Mono output
            samplerate=self.sample_rate,
            callback=self.audio_callback
        )
        self.stream.start()
        print("⚠️ EMERGENCY ALARM ACTIVATED! ⚠️")
    
    def stop_alarm(self):
        """Stop the emergency alarm"""
        if not self.is_playing:
            print("Alarm is not playing")
            return
        
        self.is_playing = False
        
        # Give the stream a moment to stop via the callback
        time.sleep(0.1)
        
        # Close the stream if it exists
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("Alarm deactivated.")
    
    def set_high_frequency(self, freq):
        """Set the high tone frequency"""
        self.high_frequency = max(500, min(1200, freq))
        print(f"High tone set to {self.high_frequency} Hz")
    
    def set_low_frequency(self, freq):
        """Set the low tone frequency"""
        self.low_frequency = max(300, min(800, freq))
        print(f"Low tone set to {self.low_frequency} Hz")
    
    def set_sweep_rate(self, rate):
        """Set the tone alternation rate"""
        self.sweep_rate = max(0.5, min(5, rate))
        print(f"Sweep rate set to {self.sweep_rate} Hz")
    
    def set_volume(self, vol):
        """Set the volume"""
        self.volume = max(0.0, min(1.0, vol))
        print(f"Volume set to {int(self.volume * 100)}%")
    
    def set_harmonics(self, count):
        """Set number of harmonics for harshness"""
        self.harmonics = max(1, min(5, count))
        print(f"Harmonic count set to {self.harmonics}")


def main():
    """Interactive console interface for the emergency alarm"""
    print("===== EMERGENCY WARNING ALARM SYSTEM =====")
    
    # Create an instance of the emergency alarm
    alarm = EmergencyAlarm()
    
    # Print available commands
    print("\nCommands:")
    print("  start    - Activate the emergency alarm")
    print("  stop     - Deactivate the alarm")
    print("  high N   - Set high tone to N Hz (500-1200)")
    print("  low N    - Set low tone to N Hz (300-800)")
    print("  sweep N  - Set tone alternation rate to N Hz (0.5-5)")
    print("  harm N   - Set harmonic count to N (1-5)")
    print("  vol N    - Set volume to N% (0-100)")
    print("  quit     - Exit the program")
    
    try:
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "start":
                alarm.start_alarm()
            elif cmd == "stop":
                alarm.stop_alarm()
            elif cmd.startswith("high "):
                try:
                    freq = float(cmd.split()[1])
                    alarm.set_high_frequency(freq)
                except (ValueError, IndexError):
                    print("Invalid frequency value. Use a number between 500 and 1200.")
            elif cmd.startswith("low "):
                try:
                    freq = float(cmd.split()[1])
                    alarm.set_low_frequency(freq)
                except (ValueError, IndexError):
                    print("Invalid frequency value. Use a number between 300 and 800.")
            elif cmd.startswith("sweep "):
                try:
                    rate = float(cmd.split()[1])
                    alarm.set_sweep_rate(rate)
                except (ValueError, IndexError):
                    print("Invalid sweep rate value. Use a number between 0.5 and 5.")
            elif cmd.startswith("harm "):
                try:
                    count = int(cmd.split()[1])
                    alarm.set_harmonics(count)
                except (ValueError, IndexError):
                    print("Invalid harmonic count. Use a number between 1 and 5.")
            elif cmd.startswith("vol "):
                try:
                    vol = float(cmd.split()[1]) / 100.0
                    alarm.set_volume(vol)
                except (ValueError, IndexError):
                    print("Invalid volume value. Use a number between 0 and 100.")
            elif cmd == "quit":
                alarm.stop_alarm()
                print("Exiting emergency alarm system...")
                break
            else:
                print("Unknown command. Try: start, stop, high N, low N, sweep N, harm N, vol N, or quit")
    except KeyboardInterrupt:
        print("\nEmergency system shutdown...")
    finally:
        # Make sure we clean up
        alarm.stop_alarm()


if __name__ == "__main__":
    main()