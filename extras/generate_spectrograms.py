import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

# Parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000
TARGET_DURATION = 5.0
TARGET_SHAPE = (256, 256)

def load_and_process_audio(file_path, target_duration=TARGET_DURATION):
    """Load and process audio file to target duration."""
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Calculate target length in samples
    target_length = int(target_duration * sr)
    
    if len(audio) < target_length:
        # If audio is shorter, replicate it
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)
    
    # Take center portion if longer than target
    if len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    
    return audio, sr

def create_mel_spectrogram(audio, sr):
    """Create mel spectrogram with specified parameters."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to target shape
    mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=TARGET_SHAPE[1], axis=1)
    mel_spec_resized = librosa.util.fix_length(mel_spec_resized, size=TARGET_SHAPE[0], axis=0)
    
    return mel_spec_resized

def save_spectrogram(spec, output_path):
    """Save spectrogram as image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()

def main():
    # Create output directory for spectrograms
    output_dir = Path('spectrograms')
    output_dir.mkdir(exist_ok=True)
    
    # Read CSV file (assuming it has columns: filename, label)
    # Modify this part according to your CSV structure
    df = pd.read_csv('your_labels.csv')  # Replace with your CSV filename
    
    # Process each unique label
    for label in df['label'].unique():
        # Get first file for this label
        sample_file = df[df['label'] == label]['filename'].iloc[0]
        
        # Process audio
        audio, sr = load_and_process_audio(sample_file)
        
        # Create spectrogram
        mel_spec = create_mel_spectrogram(audio, sr)
        
        # Save spectrogram
        output_path = output_dir / f'{label}_spectrogram.png'
        save_spectrogram(mel_spec, output_path)
        
        print(f'Generated spectrogram for label: {label}')

if __name__ == "__main__":
    main() 