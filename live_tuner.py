import sys
import numpy as np
import sounddevice as sd
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# --- CONFIGURATION ---
SAMPLE_RATE = 22050 
HOP_SIZE = 2048       
WINDOW_LENGTH = 43844 

# SENSITIVITY SETTINGS
THRESHOLD = 0.4            # AI Confidence (0.1 - 0.9)
MIN_VOLUME_THRESHOLD = 0.01 # <--- NEW: Noise Gate (0.005 is sensitive, 0.05 is strict)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_note_name(midi_number):
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    return f"{NOTE_NAMES[note_index]}{octave}"

def select_microphone():
    print("\n--- Available Audio Devices ---")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{i}] {device['name']}")
    try:
        selection = input("\nEnter Mic ID (Enter for default): ")
        return int(selection) if selection != "" else None
    except:
        return None

def main():
    device_id = select_microphone()
    print("\nLoading Model...")
    try:
        model = Model(ICASSP_2022_MODEL_PATH)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Success! Listening...")
    
    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    last_detected_notes = set()

    def callback(indata, frames, time, status):
        nonlocal audio_buffer, last_detected_notes
        if status: print(status, file=sys.stderr)

        new_data = indata.astype(np.float32)
        
        # 1. CALCULATE VOLUME
        volume = np.sqrt(np.mean(new_data**2))
        vol_bar = "|" * int(volume * 100) # Increased visual sensitivity
        
        # 2. NOISE GATE CHECK
        # If the volume is too low, clear the output and DO NOT run the AI
        if volume < MIN_VOLUME_THRESHOLD:
            print(f"Vol: {vol_bar:<10}  |  ... (Silence)", end="\r")
            return

        if new_data.shape[0] > WINDOW_LENGTH: return 

        audio_buffer = np.roll(audio_buffer, -frames, axis=1)
        audio_buffer[0, -frames:, :] = new_data

        try:
            output = model.predict(audio_buffer)
        except:
            return

        if isinstance(output, list):
            probs = output[0]
        else:
            probs = output['note']

        if probs is None: return

        probs = np.squeeze(probs) 
        max_probs = np.max(probs, axis=0) 
        
        # 3. POLYPHONIC LOGIC + HARMONIC CLEANING
        candidates = np.where(max_probs > THRESHOLD)[0]
        candidates = sorted(candidates)
        final_notes = []

        for note_idx in candidates:
            current_prob = max_probs[note_idx]
            octave_down = note_idx - 12
            
            is_ghost = False
            if octave_down in candidates:
                parent_prob = max_probs[octave_down]
                if parent_prob > (current_prob * 0.5):
                    is_ghost = True
            
            if not is_ghost:
                final_notes.append(note_idx)

        current_notes = set()
        for index in final_notes:
            midi_num = index + 21
            current_notes.add(midi_to_note_name(midi_num))

        sorted_notes = sorted(list(current_notes))
        note_str = f"♪ {', '.join(sorted_notes)}" if sorted_notes else "..."
        
        print(f"Vol: {vol_bar:<10}  |  {note_str:<40}", end="\r")
        last_detected_notes = current_notes

    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=22050, 
                            blocksize=2048, callback=callback):
            while True: sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()