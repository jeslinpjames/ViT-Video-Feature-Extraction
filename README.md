# ViT-Video-Feature-Extraction

This repository contains scripts for extracting keyframes from video files, extracting features using a Vision Transformer (ViT) model, and utilizing a Long Short-Term Memory (LSTM) network for classification.

## Keyframe Extraction (`key_frame_extraction.py`)

### Overview

The `key_frame_extraction.py` script extracts keyframes from video files. Keyframes are sampled from the video, either by duplicating frames for videos with fewer frames than required or by extracting exactly `n` keyframes for larger videos.

### Usage

1. Set the `video_path` variable in the script to the path of your video file.
2. Run the script:

    ```bash
    python key_frame_extraction.py
    ```

## Vision Transformer Feature Extraction (`image_feature_extraction_with_ViT.py`)

### Overview

The `image_feature_extraction_with_ViT.py` script extracts features from image frames using a pre-trained Vision Transformer (ViT) model. The script utilizes the `timm` library for model creation.

### Usage

1. Set the `path` variable in the script to the path of your image file.
2. Adjust the `image_size` variable as needed.
3. Run the script:

    ```bash
    python image_feature_extraction_with_ViT.py
    ```

## LSTM Classification (`lstm.py`)

### Overview

The `lstm.py` script uses an LSTM network for classification based on features extracted from keyframes. It loads features from CSV files, preprocesses the data, builds an LSTM model, trains the model, evaluates its performance, and saves the model for future use.

### Usage

1. Ensure CSV files with extracted features are available in the specified `folder_path`.
2. Run the script:

    ```bash
    python lstm.py
    ```

### Requirements

- Python 
- Libraries: `numpy`, `pandas`, `keras`, `scikit-learn`, `matplotlib`, `seaborn`, `timm`


You can install the required libraries using pip:

```bash
pip install keras opencv-python numpy matplotlib seaborn pandas scikit-learn timm

