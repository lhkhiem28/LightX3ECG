
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from imports import *
warnings.filterwarnings("ignore")

from utils import config

class Explainer():
    def __init__(self, 
        ckp_path, 
        randomized = False, 
    ):
        self.config = config(
        )
        self.model = torch.load(ckp_path, map_location = "cuda")
        if randomized:
            nn.init.normal_(self.model.classifier.weight), nn.init.normal_(self.model.classifier.bias)
        self.model.eval()

    def get_ecg(self, ecg_file):
        ecg = np.load(ecg_file)[self.config.ecg_leads, :]
        ecg = fix_length(ecg, self.config.ecg_length)
        return ecg

    def get_attention_scores(self):
        attention_scores = self.model(self.ecg, return_attention_scores = True)[1]
        attention_scores = attention_scores.squeeze(0).detach().cpu().numpy()
        return attention_scores

    def get_explanation(self, 
        ecg_file, ecg_target, 
        ecg_viz_length = 1500, save_viz_path = None, 
        show = False, 
    ):
        self.ecg = self.get_ecg(ecg_file)
        self.ecg = torch.tensor(self.ecg).float().unsqueeze(0).cuda()

        self.attention_scores = self.get_attention_scores()
        explanation = [
            attr.LayerAttribution.interpolate(attr.LayerGradCam(self.model, self.model.backbone_0.stage_3).attribute(self.ecg, target = ecg_target), (self.config.ecg_length))[0][0].detach().cpu().numpy(), 
            attr.LayerAttribution.interpolate(attr.LayerGradCam(self.model, self.model.backbone_1.stage_3).attribute(self.ecg, target = ecg_target), (self.config.ecg_length))[0][0].detach().cpu().numpy(), 
            attr.LayerAttribution.interpolate(attr.LayerGradCam(self.model, self.model.backbone_2.stage_3).attribute(self.ecg, target = ecg_target), (self.config.ecg_length))[0][0].detach().cpu().numpy(), 
        ]
        pyplot.figure(figsize = (25, 15))
        for i, cam in enumerate(explanation):
            normalized_lead, weighted_normalized_cam = minmax_normalize(self.ecg.squeeze(0)[i, :].detach().cpu().numpy()), minmax_normalize(np.abs(cam))*self.attention_scores[i]

            pyplot.subplot(len(self.config.ecg_leads), 1, i + 1), pyplot.axis("off"), pyplot.plot(normalized_lead[500:500 + ecg_viz_length], alpha = 0.2)
            pyplot.scatter(
                list(range(ecg_viz_length)), 
                normalized_lead[500:500 + ecg_viz_length], weighted_normalized_cam[500:500 + ecg_viz_length]
                , cmap = "magma"
            )
            pyplot.colorbar()
            pyplot.title(label = "ECG lead:{} - score:{:.2f}".format(self.config.ecg_leads[i], self.attention_scores[i]), loc = "left", fontdict = {"size": 16})

        if save_viz_path is not None:
            pyplot.savefig(save_viz_path, bbox_inches = "tight")
        if not show:
            pyplot.close()
        return explanation