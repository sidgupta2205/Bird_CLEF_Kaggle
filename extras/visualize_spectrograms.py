import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa.display
import librosa
from tqdm import tqdm

def visualize_spectrograms(spectrograms_dict, n_cols=4):
    """
    Visualize mel spectrograms from a dictionary where each label has a single spectrogram.
    
    Args:
        spectrograms_dict (dict): Dictionary with labels as keys and single spectrograms as values
        n_cols (int): Number of columns in the grid
    """
    # Calculate grid dimensions
    n_rows = (len(spectrograms_dict) + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(20, 5*n_rows))
    plt.suptitle("Mel Spectrograms by Label", fontsize=16, y=1.02)
    
    for idx, (label_name, mel_spec) in enumerate(tqdm(spectrograms_dict.items(), desc="Processing labels")):
        # Create subplot
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Plot spectrogram
        librosa.display.specshow(
            mel_spec,
            y_axis='mel',
            x_axis='time',
            sr=22050,  # Standard sample rate
            cmap='magma'
        )
        
        # Add colorbar
        plt.colorbar(format='%+2.0f dB')
        
        # Set title
        plt.title(label_name)
        
        # Remove axis labels for cleaner look
        plt.xlabel('')
        plt.ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def create_summary_visualization(spectrograms_dict, n_samples=1):
    """
    Create a summary visualization showing one sample from each label.
    
    Args:
        spectrograms_dict (dict): Dictionary with labels as keys and lists of spectrograms as values
        n_samples (int): Number of samples to show per label
    """
    # Calculate grid dimensions
    n_cols = 4
    n_rows = (len(spectrograms_dict) + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(20, 5*n_rows))
    plt.suptitle("Mel Spectrograms Summary", fontsize=16, y=1.02)
    
    for idx, (label_name, spectrograms) in enumerate(tqdm(spectrograms_dict.items(), desc="Creating summary")):
        if not spectrograms:
            continue
            
        # Take first spectrogram
        mel_spec = spectrograms[0]
        
        # Create subplot
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Plot spectrogram
        librosa.display.specshow(
            mel_spec,
            y_axis='mel',
            x_axis='time',
            sr=22050,
            cmap='magma'
        )
        
        # Add colorbar
        plt.colorbar(format='%+2.0f dB')
        
        # Set title
        plt.title(label_name)
        
        # Remove axis labels for cleaner look
        plt.xlabel('')
        plt.ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Example usage with a dictionary of single spectrograms
    # spectrograms_dict = {
    #     "label1": spectrogram1,
    #     "label2": spectrogram2,
    #     ...
    # }
    
    # Visualize all spectrograms in a grid
    # visualize_spectrograms(spectrograms_dict, n_cols=4)
    
    # Create summary visualization
    # create_summary_visualization(spectrograms_dict, n_samples=1) 