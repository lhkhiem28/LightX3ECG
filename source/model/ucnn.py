
from imports import *
from .cnn import *

class USEResNet18(nn.Module):
    def __init__(self, 
        lightweight, 
    ):
        super(USEResNet18, self).__init__()
        if lightweight:
            self.name = "LightUSEResNet18"
        else:
            self.name = "USEResNet18"