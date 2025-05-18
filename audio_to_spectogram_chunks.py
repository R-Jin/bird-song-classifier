import os
import random
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ====== MODIFIED ====== Import HDF5 Module
import h5py

def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_ogg_file(file_path):
    """Load OGG audio file"""
    return AudioSegment.from_ogg(file_path)

def trim_audio(audio, start_ms, end_ms):
    """Extract specific time segment from audio"""
    return audio[start_ms:end_ms]

def save_audio_segment(segment, output_path):
    """Save audio segment to file"""
    segment.export(output_path, format="ogg")

def chunk_audio_fixed(audio, chunk_duration_ms, num_chunks, output_dir, start_offset_ms=0):
    """
    chunk audio at fixed intervals
    """
    ensure_dir(output_dir)
    audio_duration = len(audio)

    for i in range(num_chunks):
        start = start_offset_ms + i * chunk_duration_ms
        end = start + chunk_duration_ms

        if end > audio_duration:
            max_start = audio_duration - chunk_duration_ms
            if max_start <= 0:
                random_start = random.randint(0, max(0, audio_duration - 1))
                segment = audio[random_start:random_start+chunk_duration_ms]
            else:
                random_start = random.randint(0, max_start)
                segment = audio[random_start:random_start+chunk_duration_ms]
        else:
            segment = audio[start:end]

        output_path = os.path.join(output_dir, f"chunk_{i+1:02d}.ogg")
        save_audio_segment(segment, output_path)

    print(f"Saved {num_chunks} chunk to {output_dir}")

# ====== MODIFIED ====== New added function to generate spectrogram for ML purpose
def generate_ml_spectrogram(audio_path, output_path):
    """Generate machine learning-ready spectrogram data"""
    y, sr = librosa.load(audio_path)
    y = librosa.resample(y, orig_sr=32000, target_sr=16000)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Save as HDF5 Format
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('spectrogram', data=D, dtype='float32', compression='gzip')
        hf.create_dataset('sr', data=sr, dtype='float32')
    print(f"ML spectrogram saved to {output_path}")

def generate_spectrogram(audio_path, output_path, figsize=(10, 4), dpi=100):
    """
    Generate and save spectrogram (保留可视化功能)
    """
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=figsize, dpi=dpi)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visual spectrogram saved to {output_path}")

# ====== MODIFIED ====== New added batch generation for ML Spectrogram
def batch_generate_ml_spectrograms(input_dir, output_dir):
    """Batch process ML spectrograms"""
    ensure_dir(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.ogg'):
            audio_path = os.path.join(input_dir, filename)
            h5_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.h5")
            generate_ml_spectrogram(audio_path, h5_path)

def batch_generate_spectrograms(input_dir, output_dir):
    """
    Batch generate spectrograms (保留可视化)
    """
    ensure_dir(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.ogg'):
            audio_path = os.path.join(input_dir, filename)
            spectrogram_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            generate_spectrogram(audio_path, spectrogram_path)
import shutil
def process_audio_file(input_file, base_output_dir):
    """Main processing function"""
    audio = load_ogg_file(input_file)

    # Trim to 1m2.5s
    target_duration = 62.5 * 1000  # 62.5 seconds in ms
    trimmed_audio = trim_audio(audio, 0, min(len(audio), target_duration))

    # Method 1: Start at 0s
    chunk_duration = 5 * 1000  # 5s in ms
    output_dir1 = os.path.join(base_output_dir, "chunks")
    chunk_audio_fixed(trimmed_audio, chunk_duration, 12, output_dir1, 0)

    # ====== MODIFIED ====== New Added for Spectrogram Generation for ML Purpose
    # Generate Spectrogram for ML Purpose
    spec_dir1_ml = os.path.join(base_output_dir)
    batch_generate_ml_spectrograms(output_dir1, spec_dir1_ml)
    shutil.rmtree(output_dir1)

import matplotlib.pyplot as plt

def plot_spectrogram(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        spec = hf['spectrogram'][:]
        sr = hf.attrs['sr']
        
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spec, 
                           sr=sr,
                           hop_length=512,
                           x_axis='time',
                           y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Validation')
    plt.show()

if __name__ == "__main__":
    # plot_spectrogram("path/to/spectrogram.h5")
    df = pd.read_csv("./birdclef-2025/train.csv")
    filenames = df['filename']
    f = filenames[0]
    drive_path = "./birdclef-2025/train_spectograms"
    ensure_dir(drive_path)
    for filename in filenames:
        directory, name = filename.split("/")
        name, _ = name.split(".")

        input_file = f"birdclef-2025/train_audio/{filename}"  

        output_dir = os.path.join(drive_path, directory, name)
        if os.path.isdir(output_dir):
            continue

        ensure_dir(output_dir)

    # # Process the file
        process_audio_file(input_file, output_dir)

        print(f"All processing complete! Results saved to directory: {output_dir}")