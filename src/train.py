import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import PersonAttributeModel
from preprocess import PersonAttributeDataset, transform

# Define paths
images_dir = 'src/VRL_challenge_PAR/images'
labels_file = 'src/VRL_challenge_PAR/train.txt'
label_names_file = 'src/VRL_challenge_PAR/label.txt'
num_attributes = len(pd.read_csv(label_names_file, header=None))

# Create dataset and dataloaders
dataset = PersonAttributeDataset(images_dir, labels_file, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = PersonAttributeModel(num_attributes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Validation
        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Calculate and print mean label accuracy
        val_labels, val_preds = evaluate_model(model, val_loader)
        accuracy_per_label, mean_label_accuracy = calculate_accuracy(val_labels, val_preds)
        print("Mean Label Accuracy:", mean_label_accuracy)

    torch.save(model.state_dict(), 'models/attribute_model.pth')
    print("Model saved successfully.")

# Evaluation function
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            all_labels.append(labels.cpu())
            all_preds.append(outputs.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    if criterion:
        running_loss = 0.0
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
        val_loss = running_loss / len(data_loader.dataset)
        return val_loss

    return all_labels, all_preds

# Calculate mean label accuracy
def calculate_accuracy(labels, preds, threshold=0.5):
    preds = preds > threshold
    correct = (preds == labels).float()
    accuracy_per_label = correct.mean(dim=0).cpu().numpy()
    mean_label_accuracy = accuracy_per_label.mean()
    return accuracy_per_label, mean_label_accuracy

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model on validation set
    val_labels, val_preds = evaluate_model(model, val_loader)
    # Calculate mean label accuracy on validation set
    accuracy_per_label, mean_label_accuracy = calculate_accuracy(val_labels, val_preds)
    print("Mean Label Accuracy:", mean_label_accuracy)
