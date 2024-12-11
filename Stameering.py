import os
import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import threading

stop_recording = threading.Event()

def record_audio(output_filename, sample_rate=16000, chunk_size=1024):
    global stop_recording
    format = pyaudio.paInt16
    channels = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)
    frames = []
    print("RECORDING START")
    
    def record():
        while not stop_recording.is_set():
            data = stream.read(chunk_size)
            frames.append(data)
    
    record_thread = threading.Thread(target=record)
    record_thread.start()
    input()  
    stop_recording.set()  
    record_thread.join()  
    print("RECORDING STOP")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def recognize_speech(audio_filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filename) as source:
        audio = recognizer.record(source)
    try:
        print("Recognizing speech...")
        transcript = recognizer.recognize_google(audio)
        print("Transcript:", transcript)
        return transcript
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def read_wave(filename):
    with wave.open(filename, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_data = wf.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)
        return audio_data, sample_rate

def extract_audio_features(filename):
    y, sr = read_wave(filename)
    duration = len(y) / sr
    silence_duration = np.sum(np.abs(y) < 0.01) / sr
    pitch = np.mean(np.abs(y))
    features = {
        'duration': duration,
        'silence_duration': silence_duration,
        'pitch': pitch
    }
    return features

def get_stammering_features_from_text(transcription):
    stammering_features = {"repetitions": 0, "prolongations": 0, "blocks": 0, "total_syllables": 0}
    words = transcription.split()
    prev_word = None
    prev_char = None
    for word in words:
        syllables = len(word) // 2
        stammering_features["total_syllables"] += syllables
        if word == prev_word:
            stammering_features["repetitions"] += 1
        if len(word) > 1:
            for char in word:
                if char == prev_char:
                    stammering_features["prolongations"] += 1
                prev_char = char
        prev_word = word
    return stammering_features

def main():
    output_directory = "D:\GURMEJ PROJECT\RECORDS"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_filename = os.path.join(output_directory, "output.wav")
    record_audio(output_filename, sample_rate=16000, chunk_size=1024)
    transcript = recognize_speech(output_filename)
    if transcript:
        stammering_features = get_stammering_features_from_text(transcript)
        audio_features = extract_audio_features(output_filename)
        print(f"Audio Features: {audio_features}")
        print(f"Stammering Features: {stammering_features}")

if __name__ == "__main__":
    main()