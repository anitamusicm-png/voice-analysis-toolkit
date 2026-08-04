# Voice Analysis Toolkit

Audio voice characteristic extraction tool for mixing, voice processing, and voice cloning preparation.

## Features

- **Pitch Metrics** — Fundamental frequency (F0), pitch stability, and range
- **Spectral Analysis** — Brightness, rolloff, and flatness
- **Dynamic Range** — Peak, mean, and min loudness
- **Clarity Score** — Harmonic-to-noise ratio (0-1 scale)

## Installation

```bash
pip install librosa scipy numpy
```

## Usage

```bash
python voice_analyzer.py path/to/vocal.wav
python voice_analyzer.py path/to/vocal.wav --output analysis.json
```

## Applications

- Pre-mixing analysis to inform EQ and compression decisions
- Voice cloning preparation (clarity and spectral assessment)
- Quality assurance on vocal takes

## Author

Ana Maidana | Recording & Mixing Engineer
