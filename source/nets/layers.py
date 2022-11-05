
import os, sys
from libs import *

class DSConv1d(nn.Module):
    def __init__(self, 
        in_channels, out_channels, 
        kernel_size, padding = 0, stride = 1, 
    ):
        super(DSConv1d, self).__init__()
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels, 
            kernel_size = kernel_size, padding = padding, stride = stride, 
            groups = in_channels, 
            bias = False, 
        )
        self.pw_conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size = 1, 
            bias = False, 
        )

    def forward(self, 
        input, 
    ):
        output = self.dw_conv(input)
        output = self.pw_conv(output)

        return output