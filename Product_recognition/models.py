import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List

class ResNet18(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size, include_top=True):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features

        if include_top:
            self.resnet18.fc = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                nn.Softmax()
            )

    def forward(self, x):
        x = self.resnet18(x)
        return x

class MobileNet(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, classCount, include_top=True):
        super(MobileNet, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=include_top)
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, classCount),
            # nn.Sigmoid()
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x



