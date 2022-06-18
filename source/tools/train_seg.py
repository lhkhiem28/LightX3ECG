
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from imports import *
warnings.filterwarnings("ignore")

lightning.seed_everything(22)
from utils import config
from data_seg import ECGDatasetSeg
from model.ucnn import USEResNet18
from engines_seg import train_fn

parser = argparse.ArgumentParser()
parser.add_argument("--multilabel", action = "store_true")
parser.add_argument("--lightweight", action = "store_true"), parser.add_argument("--use_demographic", action = "store_true"), 
parser.add_argument("--num_gpus", type = int, default = 1)
args = parser.parse_args()
config = config(
    is_multilabel = args.multilabel, 
    num_gpus = args.num_gpus, 
)

loaders = {
    "train": torch.utils.data.DataLoader(
        ECGDatasetSeg(
            config, 
            df_path = "../datasets/pretraining/Lobachevsky/train.csv", data_path = "../../datasets/pretraining/Lobachevsky/train"
        ), 
        num_workers = 8, batch_size = 32
        , shuffle = True
    ), 
    "val": torch.utils.data.DataLoader(
        ECGDatasetSeg(
            config, 
            df_path = "../datasets/pretraining/Lobachevsky/val.csv", data_path = "../../datasets/pretraining/Lobachevsky/val"
        ), 
        num_workers = 8, batch_size = 32
        , shuffle = False
    ), 
}

model = USEResNet18(args.lightweight)

optimizer = optim.Adam(
    model.parameters(), 
    lr = 5e-4, weight_decay = 5e-5, 
)
save_ckp_path = "../ckps/{}/{}".format("pretraining/Lobachevsky", model.name)
if not os.path.exists(save_ckp_path):
    os.makedirs(save_ckp_path)
train_fn(
    config, 
    loaders, model, 
    num_epochs = 40, 
    optimizer = optimizer, 
    scheduler = None, 
    save_ckp_path = save_ckp_path, training_verbose = True, 
)