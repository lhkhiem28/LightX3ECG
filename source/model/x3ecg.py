
import os, sys
from imports import *
from .cnn import *

class X3ECG(nn.Module):
    def __init__(self, 
        lightweight, use_demographic, 
        num_classes, 
    ):
        super(X3ECG, self).__init__()
        if lightweight:
            self.name = "LightX3ECG"
        else:
            self.name = "X3ECG"
        self.use_demographic = use_demographic

        self.backbone_0 = SEResNet18(lightweight)
        self.backbone_1 = SEResNet18(lightweight)
        self.backbone_2 = SEResNet18(lightweight)
        self.attention = nn.Sequential(
            nn.Linear(base_channels*24, base_channels*8), 
            nn.BatchNorm1d(base_channels*8), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(base_channels*8, 3), 
        )
        self.regressor = nn.Linear(base_channels*8, 1)

        if self.use_demographic:
            self.name += "pp"
            self.mlp = nn.Sequential(
                nn.Linear(11, base_channels*2), 
                nn.BatchNorm1d(base_channels*2), 
                nn.ReLU(), 
                nn.Dropout(0.3), 
                nn.Linear(base_channels*2, base_channels*2), 
                nn.BatchNorm1d(base_channels*2), 
                nn.ReLU(), 
            )
            self.last_drop = nn.Dropout(0.3)
            self.classifier = nn.Linear(base_channels*8 + base_channels*2, num_classes)
        else:
            self.last_drop = nn.Dropout(0.3)
            self.classifier = nn.Linear(base_channels*8, num_classes)

    def forward(self, input, return_attention_scores = False):
        feature_0 = self.backbone_0(input[0][:, 0, :].unsqueeze(1)).squeeze(2)
        feature_1 = self.backbone_1(input[0][:, 1, :].unsqueeze(1)).squeeze(2)
        feature_2 = self.backbone_2(input[0][:, 2, :].unsqueeze(1)).squeeze(2)
        attention_scores = torch.sigmoid(self.attention(torch.cat([
            feature_0, 
            feature_1, 
            feature_2, 
        ], dim = 1)))
        merged_feature = torch.sum(torch.stack([
            feature_0, 
            feature_1, 
            feature_2, 
        ], dim = 1)*attention_scores.unsqueeze(-1), dim = 1)
        sub_output = self.regressor(merged_feature).squeeze(-1)

        if self.use_demographic:
            merged_feature = self.last_drop(torch.cat([merged_feature, self.mlp(input[1])], axis = 1))
            output = self.classifier(merged_feature)
        else:
            merged_feature = self.last_drop(merged_feature)
            output = self.classifier(merged_feature)
        if not return_attention_scores:
            return (output, sub_output)
        else:
            return (output, sub_output), attention_scores