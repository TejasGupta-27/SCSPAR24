# Person Attribute Recognition

## Overview

Person Attribute Recognition is a computer vision project aimed at recognizing various attributes of individuals in images. These attributes include clothing color, type, sleeves length, carrying items, pose, view, etc. The project involves preprocessing image data, training a deep learning model, and performing inference to predict attributes in unseen images.

## Project Structure
The project is structured as follows:

- **src/**: Contains the source code and data files.
  - **extract.py**: Script to extract the dataset from ZIP files.
  - **preprocess.py**: Script for preprocessing the dataset and defining custom PyTorch datasets.
  - **model.py**: Defines the neural network model architecture.
  - **train.py**: Script to train the model using PyTorch.
  - **inference.py**: Script to perform inference using the trained model.
- **models/**: Directory to store trained model weights.
- **env.yml**: YAML file specifying the environment setup.
- **README.md**: This file.

## Execution Commands
To run different parts of the project, follow these commands:

1. **Extract Dataset**:
   ```bash
   python src/extract.py
   This script extracts the dataset from ZIP files into the src/ directory.

2. **Preprocess Data and Train Model:**:
   python src/preprocess.py
   python src/model.py
   python src/train.py 

This script preprocesses the dataset, trains the model, and saves the trained weights in the models/ directory.

3. **Perform Inference:**:
   python src/inference.py

4. **Visualization**
    python src/visualize.py
   
# Environment Setup
To create the environment, use the provided env.yml file:

bash
conda env create -f env.yml

Then, activate the environment:

conda activate person_attribute_env

# Resources Used
**Platform**: Windows/Linux
**CPUs/GPUs**: Specify the number and type of CPUs/GPUs used.
**Memory:** Specify the memory available.

**Approach**

**Dataset Extraction and Preprocessing**
The dataset is first extracted from ZIP files using extract.py. Image preprocessing, including resizing and normalization, is performed using preprocess.py. Custom PyTorch datasets are defined to load and preprocess the data efficiently.

**Model Architecture**
The neural network model architecture is defined in model.py. It utilizes a pretrained ResNet-50 backbone with a custom fully connected layer to predict the attributes.

**Training**
Model training is performed in train.py using PyTorch. The dataset is split into training and validation sets, and the model is trained using the Adam optimizer and BCEWithLogitsLoss.

**Mean Label Accuracy:** 0.8779

**Inference**
Inference on unseen images is conducted using the trained model in inference.py. The model weights are loaded, and predictions are made on the test dataset.



**Uniqueness/Novelty**
Utilization of a pretrained ResNet-50 backbone combined with a custom fully connected layer for attribute prediction.
Custom PyTorch datasets and efficient preprocessing techniques to handle large-scale image datasets.
Use of BCEWithLogitsLoss for multi-label classification.


                  +-------------------+
                  |                   |
                  |    Extract Zip    |
                  |                   |
                  +-------------------+
                           |
                           V
                  +-------------------+
                  |                   |
                  |   Preprocess Data |
                  |                   |
                  +-------------------+
                           |
                           V
                  +-------------------+
                  |                   |
                  |    Train Model    |
                  |                   |
                  +-------------------+
                           |
                           V
                  +-------------------+
                  |                   |
                  |   Make Predictions|
                  |                   |
                  +-------------------+


## Credits
PyTorch
scikit-image
Pandas
NumPy
Matplotlib
TQDM
OpenCV