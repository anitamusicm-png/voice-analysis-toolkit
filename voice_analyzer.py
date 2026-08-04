"""
voice_analyzer.py
Audio voice characteristic extraction tool for mixing and voice processing decisions.
"""

import os
import json
import numpy as np
import librosa
import librosa.display
from scipy import signal
from scipy.fft import fft
import argparse
from pathlib import Path


class VoiceAnalyzer:
    """Analyze vocal characteristics from audio files."""
    
    def __init__(self, audio_path, sr=44100):
        self.audio_path = audio_path
        self.sr = sr
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
    
    def get_pitch_metrics(self):
        """Extract fundamental frequency (F0) and pitch stability."""
        D = librosa.stft(self.y)
        S_harmonic, S_percussive = librosa.decompose.hpss(D)
        harmonic = librosa.istft(S_harmonic)
        f0 = librosa.yin(harmonic, fmin=50, fmax=400)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) == 0:
            return {"mean_f0_hz": None, "f0_stability": None, "f0_range_semitones": None}
        
        mean_f0 = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        f0_stability = 1.0 - (f0_std / mean_f0)
        f0_range_semitones = 12 * np.log2(np.max(f0_voiced) / np.min(f0_voiced))
        
        return {
            "mean_f0_hz": float(mean_f0),
            "f0_stability": float(np.clip(f0_stability, 0, 1)),
            "f0_range_semitones": float(f0_range_semitones)
        }
    
    def get_spectral_metrics(self):
        """Extract brightness and spectral shape."""
        spec_cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        spectral_centroid = np.mean(spec_cent)
        spec_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        spectral_rolloff = np.mean(spec_rolloff)
        S = np.abs(librosa.stft(self.y))
        spectral_flatness = np.mean([
            np.exp(np.mean(np.log(np.abs(S[:, t]) + 1e-10))) / (np.mean(np.abs(S[:, t])) + 1e-10)
            for t in range(S.shape[1])
        ])
        
        return {
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_rolloff_hz": float(spectral_rolloff),
            "spectral_flatness": float(np.clip(spectral_flatness, 0, 1))
        }
    
    def get_dynamic_range(self):
        """Extract loudness envelope and dynamic range."""
        S = np.abs(librosa.stft(self.y))
        rms = librosa.feature.rms(y=self.y)[0]
        rms_db = librosa.power_to_db(rms, ref=np.max)
        dynamic_range = np.max(rms_db) - np.min(rms_db)
        mean_loudness = np.mean(rms_db)
        
        return {
            "dynamic_range_db": float(dynamic_range),
            "mean_loudness_db": float(mean_loudness),
            "peak_loudness_db": float(np.max(rms_db)),
            "min_loudness_db": float(np.min(rms_db))
        }
    
    def get_clarity_score(self):
        """Estimate vocal clarity (0-1 scale)."""
        D = librosa.stft(self.y)
        S_harmonic, S_percussive = librosa.decompose.hpss(D)
        harmonic_energy = np.sum(np.abs(S_harmonic) ** 2)
        percussive_energy = np.sum(np.abs(S_percussive) ** 2)
        noise_energy = np.sum(np.abs(D) ** 2) - harmonic_energy
        hnr = harmonic_energy / (noise_energy + 1e-10)
        clarity = np.clip(hnr / (hnr + 1), 0, 1)
        
        return {"clarity_score": float(clarity)}
    
    def get_all_metrics(self):
        """Run all analyses and return comprehensive report."""
        metrics = {
            "file": os.path.basename(self.audio_path),
            "duration_seconds": float(self.duration),
            "sample_rate": self.sr,
            "pitch_metrics": self.get_pitch_metrics(),
            "spectral_metrics": self.get_spectral_metrics(),
            "dynamic_metrics": self.get_dynamic_range(),
            "clarity": self.get_clarity_score()
        }
        return metrics
    
    def save_report(self, output_path):
        """Save analysis report as JSON."""
        metrics = self.get_all_metrics()
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze vocal characteristics from audio files.")
    parser.add_argument("audio_file", help="Path to audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate (default 44100)")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ File not found: {args.audio_file}")
        return
    
    print(f"🎙️  Analyzing {args.audio_file}...")
    analyzer = VoiceAnalyzer(args.audio_file, sr=args.sr)
    metrics = analyzer.get_all_metrics()
    
    print("\n📊 VOICE ANALYSIS REPORT")
    print("=" * 50)
    print(f"File: {metrics['file']}")
    print(f"Duration: {metrics['duration_seconds']:.2f}s")
    print(f"\nPitch:")
    print(f"  Mean F0: {metrics['pitch_metrics']['mean_f0_hz']:.1f} Hz")
    print(f"  Stability: {metrics['pitch_metrics']['f0_stability']:.2f}")
    print(f"  Range: {metrics['pitch_metrics']['f0_range_semitones']:.1f} semitones")
    print(f"\nSpectral:")
    print(f"  Brightness: {metrics['spectral_metrics']['spectral_centroid_hz']:.0f} Hz")
    print(f"  Flatness: {metrics['spectral_metrics']['spectral_flatness']:.2f}")
    print(f"\nDynamic Range: {metrics['dynamic_metrics']['dynamic_range_db']:.1f} dB")
    print(f"Clarity Score: {metrics['clarity']['clarity_score']:.2f}")
    print("=" * 50)
    
    if args.output:
        analyzer.save_report(args.output)
    else:
        output_json = os.path.splitext(args.audio_file)[0] + "_analysis.json"
        analyzer.save_report(output_json)


if __name__ == "__main__":
    main()
