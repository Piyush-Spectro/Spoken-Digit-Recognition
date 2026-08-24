import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(num_classes=10, in_channels=1, device=None):
    """
    Returns a ResNet-18 model initialized from scratch (random weights)
    adapted for 1-channel Log-Mel Spectrogram input and 10 output digit classes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet-18 architecture with no pretrained weights
    model = models.resnet18(weights=None)

    # Modify conv1 layer to accept 1-channel spectrograms
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify fully-connected layer for 10 classes (digits 0-9)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)
