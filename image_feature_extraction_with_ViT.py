from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from pytorch_pretrained_vit import ViT

def extract_features_from_frame(frame, model=None):
    """
    Extracts features from an image frame using a pretrained ViT model.

    Args:
    frame (np.ndarray): The image frame as a NumPy array.
    model (ViT): Pretrained ViT model (or None to create one).

    Returns:
    torch.Tensor: Extracted features from the image.
    """
    # Load the ViT model if not provided
    if model is None:
        model = ViT('B_16_imagenet1k', pretrained=True)
        model.eval()

    # Convert the NumPy array to a PIL image
    img = Image.fromarray(frame)

    # Resize the image to match the desired input size and preprocess it
    img = transforms.Compose([
        transforms.Resize((384, 384)),  # You can change the size here
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])(img).unsqueeze(0)

    # Use the model to extract features
    with torch.no_grad():
        features = model(img)

    return features



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
