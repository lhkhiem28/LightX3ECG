
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *
from data import ECGDataset
from nets.nets import LightX3ECG
from engines import train_fn

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
parser.add_argument("--multilabel", action = "store_true")
parser.add_argument("--num_gpus", type = int, default = 1)
args = parser.parse_args()
config = {
    "ecg_leads":[
        0, 1, 
        6, 
    ], 
    "ecg_length":5000, 

    "is_multilabel":args.multilabel, 
    "device_ids":list(range(args.num_gpus)), 
}

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = "../datasets/{}/train.csv".format(args.dataset), data_path = "../../datasets/{}/train".format(args.dataset), 
            config = config, 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 224, 
        shuffle = True
    ), 
    "val":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = "../datasets/{}/val.csv".format(args.dataset), data_path = "../../datasets/{}/val".format(args.dataset), 
            config = config, 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 224, 
        shuffle = False
    ), 
}
model = LightX3ECG(
    num_classes = args.num_classes, 
)
if not config["is_multilabel"]:
    criterion = F.cross_entropy
else:
    criterion = F.binary_cross_entropy_with_logits
optimizer = optim.Adam(
    model.parameters(), 
    lr = 1e-3, weight_decay = 5e-5, 
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 1e-4, T_max = 40, 
)

save_ckp_dir = "../ckps/{}/{}".format(args.dataset, "LightX3ECG")
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, 
    model, 
    num_epochs = 70, 
    config = config, 
    criterion = criterion, 
    optimizer = optimizer, 
    scheduler = scheduler, 
    save_ckp_dir = save_ckp_dir, 
)