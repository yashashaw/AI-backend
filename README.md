LIVE TUNER CLI
Polyphonic Pitch & Duration Extraction
=====================================

Overview
--------

Live Tuner CLI is a command-line backend application that demonstrates how Spotify’s open-source AI model, Basic Pitch, can be integrated into a lightweight audio processing pipeline for polyphonic pitch detection and note duration extraction.

The tool processes audio input, detects multiple simultaneous notes (chords), computes precise onset and offset times, and outputs structured note data suitable for transcription, analysis, or downstream music systems.

The application is designed to run entirely from the terminal and function as a reusable backend service.

Entry point:

    python live_tuner.py


What Is Basic Pitch?
-------------------

Basic Pitch is an open-source Automatic Music Transcription (AMT) model released by Spotify. Unlike traditional monophonic pitch detectors, Basic Pitch supports polyphonic audio, allowing it to detect multiple overlapping notes at the same time.

Key capabilities:

- Polyphonic pitch detection
- Frame-level note activation
- Onset and offset detection
- MIDI-style note representation

Repository:
https://github.com/spotify/basic-pitch


System Architecture
-------------------

    Audio Input
        ↓
    Basic Pitch Model (Inference)
        ↓
    Note Events (pitch, onset, offset)
        ↓
    CLI Backend Parser
        ↓
    Structured Pitch + Duration Output

The Live Tuner CLI acts as a thin backend wrapper around Basic Pitch inference, handling post-processing and formatting so the extracted musical data can be easily reused.


How the CLI Backend Works
------------------------

1. Audio Input

The application accepts audio input either from a file or a live audio source (such as a microphone), depending on the implementation inside live_tuner.py.

2. Model Inference (Polyphonic Pitch Detection)

Basic Pitch processes the audio and produces:

- Frame-wise pitch activations
- Note onset timestamps
- Note offset timestamps
- MIDI pitch numbers

Because the model is polyphonic, multiple notes may be detected at the same timestamp.

3. Parsing and Post-Processing

The backend parser performs several post-processing steps:

- Converts MIDI pitch values into human-readable note names (for example: C4, F#3)
- Calculates note duration using:

      duration = offset_time - onset_time

- Groups simultaneous notes to represent polyphonic events
- Applies a noise gate and harmonic parsing logic to reduce false positives and improve accuracy

4. CLI Output

The parsed results are printed to the terminal or saved to disk in a structured format suitable for:

- Music transcription
- Live tuning feedback
- MIDI generation
- Further signal processing or analysis


Running the Program
-------------------

Requirements:

- Python 3.10.11
- Basic Pitch and its dependencies
- NumPy, PyTorch, and audio processing libraries (installed with Basic Pitch)

Install dependencies:

    pip install basic-pitch

(Alternatively, install Basic Pitch from source.)

Run the CLI application:

    python live_tuner.py

Optional flags (example):

    python live_tuner.py --input audio.wav --output notes.json

Exact flags depend on the implementation inside live_tuner.py.


Output Format (Example)
----------------------

Example JSON output:

[
  {
    "note": "C4",
    "midi": 60,
    "onset": 0.52,
    "offset": 1.03,
    "duration": 0.51
  },
  {
    "note": "E4",
    "midi": 64,
    "onset": 0.52,
    "offset": 1.03,
    "duration": 0.51
  }
]

This example demonstrates polyphonic detection, where multiple notes share the same onset and offset times.


Design Goals
------------

- CLI-first backend design
- Accurate polyphonic note detection
- Clean and readable structured output
- Extensible architecture for real-time systems, MIDI exporters, or web backends
- Suitable for integration into larger music and audio pipelines


Use Cases
---------

- Automatic music transcription
- Live tuning or pitch feedback systems
- Music Information Retrieval (MIR)
- Feeding note events into real-time or offline music processing pipelines


Credits
-------

- Basic Pitch by Spotify
- Open-source contributors to the Basic Pitch project


License
-------

This project follows the licensing terms of the Basic Pitch model and its dependencies.
Refer to the original Basic Pitch repository for full license details.
