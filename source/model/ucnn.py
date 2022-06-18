
from imports import *
from .cnn import *

class USEResNet18(nn.Module):
    def __init__(self, 
        lightweight
    ):
        super(USEResNet18, self).__init__()
        if lightweight:
            self.name = "LightUSEResNet18"
        else:
            self.name = "USEResNet18"
        self.encoder = SEResNet18(lightweight)

        self.up_3 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size = 6, padding = 2, stride = 2)
        self.destage_3 = BasicBlock(base_channels*8, base_channels*4)
        self.up_2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size = 6, padding = 2, stride = 2)
        self.destage_2 = BasicBlock(base_channels*4, base_channels*2)
        self.up_1 = nn.ConvTranspose1d(base_channels*2, base_channels*1, kernel_size = 6, padding = 2, stride = 2)
        self.destage_1 = BasicBlock(base_channels*2, base_channels*1)

        self.up_0 = nn.ConvTranspose1d(base_channels*1, base_channels*1, kernel_size = 7, padding = 3, stride = 1)
        self.destage_0 = BasicBlock(base_channels*2, base_channels*1)
        self.up_stem = nn.ConvTranspose1d(base_channels*1, base_channels*1, kernel_size = 10, padding = 3, stride = 4)
        self.destage_stem = BasicBlock(base_channels*1, 4)

    def forward(self, input):
        _, feature_list = self.encoder(input, return_feature_list = True)
        output = feature_list[-1]

        output = deep_concat(feature_list[3], self.up_3(output))
        output = self.destage_3(output)
        output = deep_concat(feature_list[2], self.up_2(output))
        output = self.destage_2(output)
        output = deep_concat(feature_list[1], self.up_1(output))
        output = self.destage_1(output)

        output = deep_concat(feature_list[0], self.up_0(output))
        output = self.destage_0(output)
        output = self.up_stem(output)
        output = self.destage_stem(output)
        return output