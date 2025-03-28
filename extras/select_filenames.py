import pandas as pd
import random
from pathlib import Path

def select_filenames_per_label(csv_path, output_dir, samples_per_label=1):
    """
    Select filenames for each label from the CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing metadata
        output_dir (str): Directory to save the selected filenames
        samples_per_label (int): Number of samples to select per label
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by primary_label
    grouped = df.groupby('primary_label')
    
    # Dictionary to store selected filenames
    selected_files = {}
    
    # Select files for each label
    for label, group in grouped:
        # Sort by rating (higher ratings first)
        group = group.sort_values('rating', ascending=False)
        
        # Select files
        selected = group.head(samples_per_label)
        selected_files[label] = selected['filename'].tolist()
        
        # Save to a text file
        output_file = output_dir / f"{label}_files.txt"
        with open(output_file, 'w') as f:
            for filename in selected_files[label]:
                f.write(f"{filename}\n")
    
    # Save summary
    summary_file = output_dir / "selection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Selection Summary:\n")
        f.write("-" * 50 + "\n")
        for label, files in selected_files.items():
            f.write(f"\nLabel: {label}\n")
            f.write(f"Number of files: {len(files)}\n")
            f.write("Files:\n")
            for file in files:
                f.write(f"  - {file}\n")
            f.write("-" * 50 + "\n")
    
    return selected_files

if __name__ == "__main__":
    # Example usage
    csv_path = "train_metadata.csv"  # Update this path to your CSV file
    output_dir = "selected_files"
    samples_per_label = 2  # Number of samples to select per label
    
    selected_files = select_filenames_per_label(csv_path, output_dir, samples_per_label)
    print(f"Selection complete. Results saved in {output_dir}") 