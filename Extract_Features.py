import os
import cv2
import numpy as np
import pandas as pd
from key_frame_extraction import extract_keyframes
from test_model import extract_features_from_frame

# Define the root folder containing subfolders with video files
root_folder = "D:/git/video_to_frame_augmentation/Kathakali dataset video/test"
output_folder = "D:/git/output_folder"  # Output folder for CSV files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all subfolders in the root folder
for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)
    output_subfolder = os.path.join(output_folder, folder)

    # Create a subfolder in the output folder for each label
    os.makedirs(output_subfolder, exist_ok=True)

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

                for idx, frame in enumerate(keyframes):
                    # Extract features from keyframes using the vision transformer
                    frame_features = extract_features_from_frame(frame)

                    # Convert tensor to a list
                    frame_features = frame_features.tolist()

                    # Save features to a CSV file for each frame
                    output_csv_file = os.path.join(output_subfolder, f"{video_file.split('.')[0]}_frame{idx}.csv")
                    df = pd.DataFrame({'X': frame_features})
                    df.to_csv(output_csv_file, index=False)
