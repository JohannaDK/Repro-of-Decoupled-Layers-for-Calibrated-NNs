import torch.nn as nn
from torchvision.models import resnet50

class ResNet50(nn.Module):
    """
    ResNet-50 architecture adapted for the given dataset.
    """
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet50, self).__init__()
        # Load the ResNet-50 backbone
        self.model = resnet50(pretrained=pretrained)
        # Modify the final fully connected layer for the specific number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)