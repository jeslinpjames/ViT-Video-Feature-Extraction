import os
import cv2
import numpy as np
from key_frame_extraction import extract_keyframes
from test_model import extract_features_from_frame

def save_features_and_labels(X, y, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'X_features.npy'), X)
    np.save(os.path.join(output_folder, 'y_labels.npy'), y)

# Define the root folder containing subfolders with video files
root_folder = "D:/git/video_to_frame_augmentation/Kathakali dataset video/test"
output_folder = "D:/git/output_folder/"


# Initialize lists to store features and labels
X = []  # Features
y = []  # Labels

# Iterate through all subfolders in the root folder
for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)
    
    # Check if it's a directory (subfolder)
    if os.path.isdir(folder_path):
        # Iterate through video files in the subfolder
        for video_file in os.listdir(folder_path):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(folder_path, video_file)

                # Extract keyframes from the video
                keyframes = extract_keyframes(video_path)
                if not keyframes:
                    print(f"No keyframes extracted from {video_path}.")
                else:
                    print(f"Extracted {len(keyframes)} keyframes from {video_path}.")

                # Extract features from keyframes using the vision transformer
                frame_features = [extract_features_from_frame(frame) for frame in keyframes]

                # Append features and labels to the lists
                X.append(frame_features)
                y.append(folder)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

save_features_and_labels(X, y, output_folder)