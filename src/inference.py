import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import PersonAttributeModel
from preprocess import TestPersonAttributeDataset, test_transform

# Load the label names
label_names_file = 'src/VRL_challenge_PAR/label.txt'
label_names = pd.read_csv(label_names_file, header=None).values.squeeze().tolist()
num_attributes = len(label_names)

# Load the trained model
model = PersonAttributeModel(num_attributes)
model.load_state_dict(torch.load('models/attribute_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define paths
test_images_dir = 'src/SCSPAR24_Testdata'
test_dataset = TestPersonAttributeDataset(test_images_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to make predictions on the test set
def make_predictions(model, data_loader, label_names):
    model.eval()
    all_preds = []
    all_image_names = []

    with torch.no_grad():
        for images, image_names in tqdm(data_loader):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            all_preds.append(outputs.cpu())
            all_image_names.extend(image_names)

    all_preds = torch.cat(all_preds).numpy()
    return all_image_names, all_preds

# Make predictions on the test set
test_image_names, test_preds = make_predictions(model, test_loader, label_names)

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(test_preds, columns=label_names)
predictions_df.insert(0, 'Image', test_image_names)

# Apply a threshold to convert probabilities to binary predictions
threshold = 0.5
binary_predictions_df = predictions_df.copy()
binary_predictions_df.iloc[:, 1:] = (binary_predictions_df.iloc[:, 1:] > threshold).astype(int)

df = binary_predictions_df.copy()

# Remove ".jpg" suffix and convert to integers for sorting
df['Image'] = df['Image'].str.replace('.jpg', '').astype(int)

# Sort the dataframe by the numerical image names
df = df.sort_values(by='Image')

# Add the ".jpg" suffix back to the image names
df['Image'] = df['Image'].astype(str) + '.jpg'

df = df.iloc[:,1:]
file_path='submission.txt'
# Write the data to the text file excluding the header row
with open(file_path, 'w') as txt_file:
    # Exclude header row and iterate over the rows in the DataFrame
    for index, row in df.iterrows():
        # Write each row to the text file
        txt_file.write(' '.join(map(str, row.values)) + '\n')