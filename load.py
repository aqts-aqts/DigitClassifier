import numpy as np
from PIL import Image
import os

def load_image(filename: str) -> np.ndarray:
    img = Image.open(filename).convert('L')
    img_array = np.array(img)
    normalized = 1 - img_array / 255.0
    return normalized.flatten()

def load_images(directory: str) -> np.ndarray:
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = load_image(os.path.join(directory, filename)).flatten()
            images.append(img)
    return np.array(images)