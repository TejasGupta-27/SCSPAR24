import torch.nn as nn
from torchvision import models

class PersonAttributeModel(nn.Module):
    def __init__(self, num_attributes):
        super(PersonAttributeModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_attributes)

    def forward(self, x):
        return self.base_model(x)
