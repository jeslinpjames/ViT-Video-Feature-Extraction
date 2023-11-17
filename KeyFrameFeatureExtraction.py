import os
import cv2
import numpy as np
import pandas as pd
from key_frame_extraction import extract_keyframes
from image_feature_extraction_with_ViT import extract_features_from_frame

# Define the root folder containing subfolders with video files
root_folder = "D:/git/video_to_frame_augmentation/Kathakali dataset video/test"
output_folder = "D:/git/output_folder"  # Output folder for CSV files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

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
                    all_features = []
                    x = 0
                    for frame in keyframes:
                        # Extract features from keyframes using the vision transformer
                        x+=1
                        frame_features = extract_features_from_frame(frame)
                        print(x,"/",len(keyframes))
                        # Append the features to the list for this subfolder
                        all_features.append(frame_features)

                    if all_features:
                        # Stack all features vertically to create a single 2D array
                        all_features = np.vstack(all_features)

                        # Create a DataFrame with columns for each feature
                        df = pd.DataFrame(data=all_features)

                        # Save the features for all frames in a single CSV file
                        output_csv_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.csv")
                        df.to_csv(output_csv_file, index=False)