import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from preprocess import PersonAttributeDataset
import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Define paths
images_dir = 'src/VRL_challenge_PAR/images'
labels_file = 'src/VRL_challenge_PAR/train.txt'
label_names_file = 'src/VRL_challenge_PAR/label.txt'
num_attributes = len(pd.read_csv(label_names_file, header=None))
# Create the dataset
dataset = PersonAttributeDataset(images_dir, labels_file, transform=transform)

# Define a function to visualize images and their corresponding labels
def visualize_dataset(dataset, label_names, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    for i in range(num_images):
        image, labels = dataset[i]
        image = image.permute(1, 2, 0).numpy()  # Convert from tensor to numpy for visualization

        axes[i].imshow(image)
        label_indices = torch.nonzero(labels).squeeze().tolist()  # Get indices of labels with value 1
        label_text = ', '.join([label_names[idx] for idx in label_indices])
        axes[i].set_title(f"Labels: {label_text}")
        axes[i].axis('off')

    plt.show()

# Load label names
label_names_file = 'src/VRL_challenge_PAR/label.txt'
with open(label_names_file, 'r') as f:
    label_names = f.read().splitlines()

# Visualize images and their labels
visualize_dataset(dataset, label_names, num_images=5)
