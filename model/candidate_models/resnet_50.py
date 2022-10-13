import torch
import torch.nn as nn
import torchvision

class ResNet50(nn.Module):
  def __init__(self, num_classes=10):
    super(ResNet50, self).__init__()
    
    self.resnet50 = torchvision.models.resnet50(pretrained=True)
    
    self.resnet50.fc = nn.LazyLinear(out_features=num_classes, bias=True)
    
  def forward(self, x):
    x = self.resnet50(x)
    return x