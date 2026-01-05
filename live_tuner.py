import sys
import time
import numpy as np
import sounddevice as sd
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# --- CONFIGURATION ---
SAMPLE_RATE = 22050 
HOP_SIZE = 2048       
WINDOW_LENGTH = 43844 
THRESHOLD = 0.4            
MIN_VOLUME_THRESHOLD = 0.01 

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

    print("Success! Logging notes... (Press Ctrl+C to stop)")
    print("-" * 50)
    
    # BUFFERS & STATE TRACKING
    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    
    # State variables for the logger
    current_instance_id = 0
    active_notes = set()       # The notes currently being logged
    last_candidate_notes = set() # For stability check (debouncing)
    stability_counter = 0      # How many frames the notes have held steady
    
    start_time_ref = time.time() # Reference time (0ms)

    def callback(indata, frames, time_info, status):
        nonlocal audio_buffer, active_notes, current_instance_id
        nonlocal last_candidate_notes, stability_counter
        
        if status: print(status, file=sys.stderr)

        new_data = indata.astype(np.float32)
        
        # 1. Volume Gate
        volume = np.sqrt(np.mean(new_data**2))
        if new_data.shape[0] > WINDOW_LENGTH: return 
        
        # Shift Buffer
        audio_buffer = np.roll(audio_buffer, -frames, axis=1)
        audio_buffer[0, -frames:, :] = new_data
        
        # If silent, assume no notes (skip AI to save CPU)
        if volume < MIN_VOLUME_THRESHOLD:
            detected_notes = set()
        else:
            # Run AI
            try:
                output = model.predict(audio_buffer)
            except:
                return

            if isinstance(output, list):
                probs = output[0]
            else:
                probs = output['note']

            if probs is None: 
                detected_notes = set()
            else:
                probs = np.squeeze(probs) 
                max_probs = np.max(probs, axis=0) 
                
                # Logic: Polyphonic + Harmonic Pruning
                candidates = np.where(max_probs > THRESHOLD)[0]
                candidates = sorted(candidates)
                final_indices = []
                for note_idx in candidates:
                    current_prob = max_probs[note_idx]
                    octave_down = note_idx - 12
                    is_ghost = False
                    if octave_down in candidates:
                        if max_probs[octave_down] > (current_prob * 0.5):
                            is_ghost = True
                    if not is_ghost:
                        final_indices.append(note_idx)

                detected_notes = set()
                for index in final_indices:
                    midi_num = index + 21
                    detected_notes.add(midi_to_note_name(midi_num))

        # --- LOGGING LOGIC WITH STABILITY CHECK ---
        
        # Only change state if the AI sees the SAME notes for 2 frames in a row
        # This prevents "Instance 5: [C]" -> "Instance 6: [C, D]" -> "Instance 7: [C]" flickering
        if detected_notes == last_candidate_notes:
            stability_counter += 1
        else:
            stability_counter = 0
            last_candidate_notes = detected_notes

        # If stable enough (approx 20-50ms stable), we accept the change
        if stability_counter >= 2: 
            
            # Check if the "Stable" notes are different from the "Logged" notes
            if detected_notes != active_notes:
                
                current_time_ms = int((time.time() - start_time_ref) * 1000)
                
                # 1. End previous instance (if it wasn't silence)
                if len(active_notes) > 0:
                    print(f"time: {current_time_ms} ms | instance {current_instance_id} ended")
                
                # 2. Start new instance (if not silence)
                if len(detected_notes) > 0:
                    current_instance_id += 1
                    sorted_notes = sorted(list(detected_notes))
                    print(f"Instance {current_instance_id}: time: {current_time_ms} ms | notes: {sorted_notes}")
                
                # 3. Update State
                active_notes = detected_notes
                
                # Add a visual separator if we went to silence
                if len(detected_notes) == 0:
                    print("-" * 20)

    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=22050, 
                            blocksize=2048, callback=callback):
            while True: sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nLog Stopped.")

if __name__ == "__main__":
    main()