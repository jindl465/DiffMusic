import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.spatial.distance import cosine

def load_audio_features(audio_path, sr=16000, duration=10, n_mels=128, n_fft=1024):
    """Load audio file and calculate primary features for the first 10 seconds."""
    if not os.path.isfile(audio_path):  
        print(f"Skipping (Not a valid file): {audio_path}")
        return None

    # Load audio limited to 10 seconds
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    y = librosa.util.fix_length(y, size=sr * duration)
    y = librosa.util.normalize(y)  # Normalize volume

    # Set hop_length to distribute 1000 frames across 10 seconds
    num_frames = 1000  
    hop_length = sr * duration // num_frames  

    # 1. Generate Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  

    target_frames = num_frames
    if mel_spec_db.shape[1] > target_frames:
        mel_spec_db = mel_spec_db[:, :target_frames]
    
    # 2. Calculate Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    return y, sr, mel_spec_db, spec_centroid


def compute_stft_similarity(y1, y2):
    """Calculate Cosine Similarity based on STFT."""
    stft1 = np.abs(librosa.stft(y1, n_fft=2048))
    stft2 = np.abs(librosa.stft(y2, n_fft=2048))
    
    # Flatten and normalize vectors
    stft1 = stft1.flatten() / np.linalg.norm(stft1)
    stft2 = stft2.flatten() / np.linalg.norm(stft2)
    
    return 1 - cosine(stft1, stft2)  # Higher values indicate higher similarity


def plot_mel_spectrogram(mel_spec, sr, output_path, title, duration=10):
    """Save Mel-Spectrogram as a square image."""
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="magma")
    plt.xlim(0, duration)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_spectral_centroid(spec_centroid_gt, spec_centroid_diff, spec_centroid_other, output_path):
    """Save Spectral Centroid comparison plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(spec_centroid_gt, label="GT", alpha=0.7, color='blue')
    plt.plot(spec_centroid_diff, label="DiffMusic", alpha=0.7, color='orange', linestyle='dashed')
    plt.plot(spec_centroid_other, label="Mozart's Touch", alpha=0.7, color='green', linestyle='dashed')
    plt.xlabel("Frames")
    plt.ylabel("Spectral Centroid (Hz)")
    plt.legend()
    plt.grid()
    plt.title("Spectral Centroid Comparison")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stft_similarity(stft_sim_1, stft_sim_2, output_path):
    """Save STFT Cosine Similarity as a bar chart."""
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["GT-Diff", "GT-Other"], [stft_sim_1, stft_sim_2], color=["red", "purple"], alpha=0.5)
    plt.ylim([0, 1])
    plt.title("STFT Cosine Similarity")
    plt.bar_label(bars, fmt="%.3f")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_comparison(gt_path, diffmusic_path, other_path, output_base_dir):
    """Compare Ground Truth vs. DiffMusic vs. Other Model and save visualizations."""
    gt_features = load_audio_features(gt_path)
    diffmusic_features = load_audio_features(diffmusic_path)
    other_features = load_audio_features(other_path)

    if any(x is None for x in [gt_features, diffmusic_features, other_features]):
        print(f"Skipping comparison due to missing data: {gt_path}")
        return  

    y_gt, sr, mel_gt, spec_centroid_gt = gt_features
    y_diff, _, mel_diff, spec_centroid_diff = diffmusic_features
    y_other, _, mel_other, spec_centroid_other = other_features

    # Calculate similarities
    stft_sim_1 = compute_stft_similarity(y_gt, y_diff)
    stft_sim_2 = compute_stft_similarity(y_gt, y_other)

    # Create individual directory for the specific file
    file_name = os.path.basename(gt_path).replace(".wav", "")
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save Mel-Spectrograms
    plot_mel_spectrogram(mel_gt, sr, os.path.join(output_dir, "mel_spectrogram_gt.png"), "GT Mel-Spectrogram")
    plot_mel_spectrogram(mel_diff, sr, os.path.join(output_dir, "mel_spectrogram_diff.png"), "DiffMusic Mel-Spectrogram")
    plot_mel_spectrogram(mel_other, sr, os.path.join(output_dir, "mel_spectrogram_other.png"), "Other Mel-Spectrogram")

    # Save Spectral Centroid Comparison
    plot_spectral_centroid(
        spec_centroid_gt, spec_centroid_diff, spec_centroid_other,
        os.path.join(output_dir, "spectral_centroid.png")
    )

    # Save Similarity Bar Chart
    plot_stft_similarity(stft_sim_1, stft_sim_2, os.path.join(output_dir, "stft_similarity.png"))


def compare_all_files(gt_dir, diffmusic_dir, other_dir, output_base_dir="output_images"):
    """Compare all matching .wav files across the three provided directories."""
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.wav')])
    diff_files = sorted([f for f in os.listdir(diffmusic_dir) if f.endswith('.wav')])
    other_files = sorted([f for f in os.listdir(other_dir) if f.endswith('.wav')])

    common_files = set(gt_files) & set(diff_files) & set(other_files)
    
    if not common_files:
        print("No matching files found across directories!")
        return

    for file_name in common_files:
        # Example condition for a specific test file
        if file_name == "1_o3JEzRwwY.wav":
            print(f"Processing: {file_name}")
            plot_comparison(
                os.path.join(gt_dir, file_name), 
                os.path.join(diffmusic_dir, file_name), 
                os.path.join(other_dir, file_name), 
                output_base_dir
            )

# Execution example
if __name__ == "__main__":
    GT_AUDIO_DIR = "/mnt/storage1/Jin/MUImage/audioset_eval_wav/"
    DIFFMUSIC_AUDIO_DIR = "/mnt/storage1/Jin/diffMusic/result/test31_muimage_best2/llm/"
    OTHER_AUDIO_DIR = "/mnt/storage1/Jin/MUImage/outputs_wav/"
    OUTPUT_DIR = "top_similar_images_best2"

    compare_all_files(GT_AUDIO_DIR, DIFFMUSIC_AUDIO_DIR, OTHER_AUDIO_DIR, OUTPUT_DIR)