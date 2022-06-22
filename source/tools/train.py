
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from imports import *
warnings.filterwarnings("ignore")

lightning.seed_everything(22)
from utils import config
from data import ECGDataset
from model.x3ecg import X3ECG
from engines import train_fn

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
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
        ECGDataset(
            config, 
            df_path = "../datasets/{}/train.csv".format(args.dataset), data_path = "../../datasets/{}/train".format(args.dataset)
            , augment = True
        ), 
        num_workers = 8, batch_size = 224
        , shuffle = True
    ), 
    "val": torch.utils.data.DataLoader(
        ECGDataset(
            config, 
            df_path = "../datasets/{}/val.csv".format(args.dataset), data_path = "../../datasets/{}/val".format(args.dataset)
            , augment = False
        ), 
        num_workers = 8, batch_size = 224
        , shuffle = False
    ), 
}

for regressor_lambda in [round(i, 2) for i in np.arange(1, 11)*0.05]:
    model = X3ECG(
        lightweight = args.lightweight, use_demographic = args.use_demographic, 
        num_classes = args.num_classes, 
    )
    # model = torch.load("../ckps/pretraining/Lobachevsky/X3ECG/best.ptl", map_location = "cuda")
    # model.classifier = nn.Linear(512, args.num_classes)
    # encoder = torch.load("../ckps/pretraining/Lobachevsky/USEResNet18/best.ptl", map_location = "cuda").encoder
    # model.backbone_0 = encoder
    # model.backbone_1 = encoder
    # model.backbone_2 = encoder

    optimizer = optim.Adam(
        model.parameters(), 
        lr = 1e-3, weight_decay = 5e-5, 
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        eta_min = 0.1*1e-3, T_max = 50, 
    )
    save_ckp_path = "../ckps/{}/{}".format(args.dataset, model.name)
    if not os.path.exists(save_ckp_path):
        os.makedirs(save_ckp_path)
    train_fn(
        config, 
        loaders, model, 
        regressor_lambda = regressor_lambda, 
        num_epochs = 100, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        save_ckp_path = save_ckp_path, training_verbose = True, 
    )