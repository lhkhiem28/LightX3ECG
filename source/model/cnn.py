
from imports import *

base_channels = 64
class SEResNet18(nn.Module):
    def __init__(self, 
        lightweight, 
    ):
        super(SEResNet18, self).__init__()
        if lightweight:
            self.name = "LightSEResNet18"
            self.block = LightSEBlock
        else:
            self.name = "SEResNet18"
            self.block = SEBlock

        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size = 15, padding = 7, stride = 2), 
            nn.BatchNorm1d(base_channels), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size = 3, padding = 1, stride = 2), 
        )
        self.stage_0 = nn.Sequential(
            self.block(base_channels), 
            self.block(base_channels), 
        )

        self.stage_1 = nn.Sequential(
            self.block(base_channels*1, downsample = True), 
            self.block(base_channels*2), 
        )
        self.stage_2 = nn.Sequential(
            self.block(base_channels*2, downsample = True), 
            self.block(base_channels*4), 
        )
        self.stage_3 = nn.Sequential(
            self.block(base_channels*4, downsample = True), 
            self.block(base_channels*8), 
        )

        self.global_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        output = self.stem(input)
        output = self.stage_0(output)

        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)

        output = self.global_pooling(output)
        return output