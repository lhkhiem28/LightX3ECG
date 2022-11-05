
import os, sys
from libs import *
from .layers import *
from .modules import *
from .bblocks import *
from .backbones import *

class LightX3ECG(nn.Module):
    def __init__(self, 
        base_channels = 64, 
        num_classes = 1, 
    ):
        super(LightX3ECG, self).__init__()
        self.backbone_0 = LightSEResNet18(base_channels)
        self.backbone_1 = LightSEResNet18(base_channels)
        self.backbone_2 = LightSEResNet18(base_channels)
        self.lw_attention = nn.Sequential(
            nn.Linear(
                base_channels*24, base_channels*8, 
            ), 
            nn.BatchNorm1d(base_channels*8), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(
                base_channels*8, 3, 
            ), 
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                base_channels*8, num_classes, 
            ), 
        )

    def forward(self, 
        input, 
        return_attention_scores = False, 
    ):
        features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
        features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)
        features_2 = self.backbone_2(input[:, 2, :].unsqueeze(1)).squeeze(2)
        attention_scores = torch.sigmoid(
            self.lw_attention(
                torch.cat(
                [
                    features_0, 
                    features_1, 
                    features_2, 
                ], 
                dim = 1, 
                )
            )
        )
        merged_features = torch.sum(
            torch.stack(
            [
                features_0, 
                features_1, 
                features_2, 
            ], 
            dim = 1, 
            )*attention_scores.unsqueeze(-1), 
            dim = 1, 
        )

        output = self.classifier(merged_features)

        if not return_attention_scores:
            return output
        else:
            return output, attention_scores