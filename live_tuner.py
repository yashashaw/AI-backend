import sys
import time
import numpy as np
import sounddevice as sd
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
HOP_SIZE = 2048         # The amount of new audio we process per step
WINDOW_LENGTH = 43844   # The context length required by the model (~2 seconds)

# Sensitivity
NOTE_THRESHOLD = 0.4        # Confidence to sustain a note
ONSET_THRESHOLD = 0.5       # Confidence to detect a NEW hit (attack)
MIN_VOLUME_THRESHOLD = 0.001 # Microphone gate (raise this if noise triggers notes)

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

def print_volume_bar(volume):
    """Prints a visual volume bar to help debug microphone levels."""
    bar_len = 30
    fill = int(min(volume * 10, 1.0) * bar_len)
    bar = '#' * fill + '-' * (bar_len - fill)
    sys.stdout.write(f"\rVolume: [{bar}] {volume:.3f} ")
    sys.stdout.flush()

def main():
    device_id = select_microphone()
    print("\nLoading Model (this may take a moment)...")
    try:
        model = Model(ICASSP_2022_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nSuccess! Logging notes...")
    print("Adjust your mic so typical playing hits 0.1 - 0.3 on the volume meter.")
    print("Press Ctrl+C to stop.\n")
    print("-" * 50)

    # BUFFERS
    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    
    # STATE TRACKING
    active_notes = {}  # Dict mapping midi_number -> start_time
    instance_id = 0
    start_time_ref = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal audio_buffer, active_notes, instance_id
        
        if status:
            print(status, file=sys.stderr)

        new_data = indata.astype(np.float32)
        
        # 1. Volume Gate & Debug
        volume = np.sqrt(np.mean(new_data**2))
        print_volume_bar(volume)

        # Shift Buffer
        audio_buffer = np.roll(audio_buffer, -frames, axis=1)
        audio_buffer[0, -frames:, :] = new_data
        
        # Skip AI processing if silence (saves CPU)
        if volume < MIN_VOLUME_THRESHOLD:
            # If we were tracking notes, clear them because of silence
            if active_notes:
                print(f"\n[Silence] All notes ended.")
                active_notes = {}
            return

        # 2. Run AI Inference
        try:
            output = model.predict(audio_buffer)
        except Exception:
            return

        # Output format is dictionary with keys: 'note', 'onset', 'contour'
        # Shape: (batch, time_frames, 88_notes)
        note_probs = output['note']
        onset_probs = output['onset']

        if note_probs is None: 
            return

        # --- KEY FIX: TEMPORAL SLICING ---
        # Instead of looking at the whole 2-second buffer, look only at the 
        # last 3 frames (approx last 30-50ms) which corresponds to the NEW audio.
        # We assume 3 frames is enough to be stable but fast enough to catch rapid notes.
        focus_window_size = 5 
        
        # Get max probability in the "Now" window
        current_notes_max = np.max(note_probs[0, -focus_window_size:, :], axis=0)
        current_onsets_max = np.max(onset_probs[0, -focus_window_size:, :], axis=0)

        # 3. Process Notes
        # We scan all 88 piano keys (MIDI 21 to 108)
        detected_this_frame = set()

        for i in range(88):
            midi_num = i + 21
            prob_note = current_notes_max[i]
            prob_onset = current_onsets_max[i]
            
            # CONDITION 1: Note is loud enough to be considered "Sustaining"
            is_sustaining = prob_note > NOTE_THRESHOLD
            
            # CONDITION 2: Note is being "Attacked" (Hit freshly)
            is_attack = prob_onset > ONSET_THRESHOLD

            if is_sustaining:
                detected_this_frame.add(midi_num)
                
                # LOGIC: If note is already active, but we detect a NEW ATTACK, restart it.
                if midi_num in active_notes and is_attack:
                    # Note re-articulation (e.g. playing the same chord twice quickly)
                    old_timestamp = active_notes[midi_num]
                    # Only re-trigger if some time has passed (debounce fast glitches, e.g., 100ms)
                    if (time.time() - old_timestamp) > 0.1:
                        print(f"\nRE-TRIGGER: {midi_to_note_name(midi_num)}")
                        active_notes[midi_num] = time.time()
                
                # LOGIC: New Note
                elif midi_num not in active_notes:
                    current_ms = int((time.time() - start_time_ref) * 1000)
                    active_notes[midi_num] = time.time()
                    print(f"\n[{current_ms}ms] Note ON: {midi_to_note_name(midi_num)}")

        # 4. Handle Note Offs
        # If a note was active but is no longer detected in the current frame
        active_ids = list(active_notes.keys())
        for midi_num in active_ids:
            if midi_num not in detected_this_frame:
                print(f"\nNote OFF: {midi_to_note_name(midi_num)}")
                del active_notes[midi_num]

    try:
        # Reduced blocksize slightly for better responsiveness (optional)
        with sd.InputStream(device=device_id, channels=1, samplerate=22050, 
                            blocksize=HOP_SIZE, callback=callback):
            while True: sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nLog Stopped.")

if __name__ == "__main__":
    main()