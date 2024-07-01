import os
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PersonAttributeDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_file, sep=' ', header=None)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, f"{self.labels.iloc[idx, 0]}.jpg")
        image = Image.open(img_name).convert("RGB")

        # Apply sharpness and contrast adjustment
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast

        # Exclude the first column which is the image number
        labels = torch.tensor(self.labels.iloc[idx, 1:].values.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class TestPersonAttributeDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        # Apply sharpness and contrast adjustment
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]
# Define transformations for the test dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
