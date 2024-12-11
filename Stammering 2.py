import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal

audio_path = r'D:\GURMEJ PROJECT\RECORDS\output.wav'

try:
    y, sr = librosa.load(audio_path, sr=None)
except Exception as e:
    print(f"Error loading audio file: {e}")
    exit(1)

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.7)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

rms = librosa.feature.rms(y=y)[0]
times = librosa.times_like(rms, sr=sr)

plt.subplot(2, 1, 2)
plt.plot(times, rms, label='RMS Energy', color='r')
plt.title('RMS Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

def analyze_audio_features(y, sr):
    duration = len(y) / sr
    silence_duration = np.sum(np.abs(y) < 0.01) / sr
    pitch = np.mean(librosa.core.pitch.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')))
    features = {
        'duration': duration,
        'silence_duration': silence_duration,
        'pitch': pitch
    }
    return features

def detect_stammering_features(y, sr):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)

    peaks, _ = signal.find_peaks(rms[0], height=np.mean(rms), distance=sr*0.01)
    troughs, _ = signal.find_peaks(-rms[0], height=-np.mean(rms), distance=sr*0.01)

    repetitions = np.sum(np.diff(peaks) < sr*0.1)
    prolongations = np.sum(np.diff(troughs) < sr*0.1)
    blocks = np.sum(rms[0] < np.mean(rms) / 2)

    stammering_features = {
        'repetitions': repetitions,
        'prolongations': prolongations,
        'blocks': blocks
    }
    return stammering_features

audio_features = analyze_audio_features(y, sr)
stammering_features = detect_stammering_features(y, sr)

print(f"Audio Features: {audio_features}")
print(f"Stammering Features: {stammering_features}")