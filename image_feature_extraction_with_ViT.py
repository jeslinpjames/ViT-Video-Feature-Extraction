import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image

def extract_features_from_frame(frame, model=None):
    """
    Extracts features from an image frame using a pretrained ViT model.

    Args:
    frame (np.ndarray): The image frame as a NumPy array.
    model (nn.Module): Pretrained ViT model (or None to create one).

    Returns:
    torch.Tensor: Extracted features from the image.
    """
    # Check if a GPU is available and move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the ViT model if not provided
    if model is None:
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = model.to(device)
        model.eval()

    # Convert the NumPy array to a PIL image
    img = Image.fromarray(frame)

    # Check the number of channels
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image to match the desired input size and preprocess it
    img = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size accordingly
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])(img).unsqueeze(0).to(device)

    # Use the model to extract features
    with torch.no_grad():
        features = model(img)

    return features.cpu()  # Move the features back to the CPU for further processing if needed


if __name__ == "__main__":
    path = "D:/git/video_to_frame_augmentation/data/Adbutham/person1/person1_1.jpeg"
    image_size = 384  # Adjust the image size to your needs
    model = ViT('B_16_imagenet1k', pretrained=True)
    model.eval()

    image = Image.open(path)
    image = np.array(image)

    # Extract features
    features = extract_features_from_frame(image, model)
    print(features)  # Print the shape of the extracted features
