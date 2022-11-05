
import os, sys
from libs import *
from utils import *

def train_fn(
    train_loaders, 
    model, 
    num_epochs, 
    config, 
    criterion, 
    optimizer, 
    scheduler = None, 
    save_ckp_dir = "./", 
    training_verbose = True, 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = config["device_ids"])

    best_f1 = 0.0
    for epoch in tqdm(range(1, num_epochs + 1), disable = training_verbose):
        if training_verbose:print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + "-"*16)

        model.train()
        running_loss = 0.0
        running_labels, running_preds = [], []
        for ecgs, labels in tqdm(train_loaders["train"], disable = not training_verbose):
            ecgs, labels = ecgs.cuda(), labels.cuda()

            logits = model(ecgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*ecgs.size(0)
            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.50, 1, 0))
            running_labels.extend(labels), running_preds.extend(preds)

        if (scheduler is not None) and (not epoch > scheduler.T_max):
            scheduler.step()

        epoch_loss, epoch_f1 = running_loss/len(train_loaders["train"].dataset), f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        if training_verbose:
            print("{:<5} - loss:{:.4f}, f1:{:.4f}".format(
                "train", 
                epoch_loss, epoch_f1
            ))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_labels, running_preds = [], []
            for ecgs, labels in tqdm(train_loaders["val"], disable = not training_verbose):
                ecgs, labels = ecgs.cuda(), labels.cuda()

                logits = model(ecgs)
                loss = criterion(logits, labels)

                running_loss = running_loss + loss.item()*ecgs.size(0)
                labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.50, 1, 0))
                running_labels.extend(labels), running_preds.extend(preds)

        epoch_loss, epoch_f1 = running_loss/len(train_loaders["val"].dataset), f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        if training_verbose:
            print("{:<5} - loss:{:.4f}, f1:{:.4f}".format(
                "val", 
                epoch_loss, epoch_f1
            ))
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1; torch.save(model.module, "{}/best.ptl".format(save_ckp_dir))

    print("\nStart Evaluation ...\n" + " = "*16)
    model = torch.load("{}/best.ptl".format(save_ckp_dir), map_location = "cuda")
    model = nn.DataParallel(model, device_ids = config["device_ids"])

    with torch.no_grad():
        model.eval()
        running_labels, running_preds = [], []
        for ecgs, labels in tqdm(train_loaders["val"], disable = not training_verbose):
            ecgs, labels = ecgs.cuda(), labels.cuda()

            logits = model(ecgs)

            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config["is_multilabel"] else list(torch.sigmoid(logits).detach().cpu().numpy())
            running_labels.extend(labels), running_preds.extend(preds)

    if config["is_multilabel"]:
        running_labels, running_preds = np.array(running_labels), np.array(running_preds)

        optimal_thresholds = thresholds_search(running_labels, running_preds)
        running_preds = np.stack([
            np.where(running_preds[:, cls] >= optimal_thresholds[cls], 1, 0) for cls in range(running_preds.shape[1])
        ]).transpose()
    val_loss, val_f1 = running_loss/len(train_loaders["val"].dataset), f1_score(
        running_labels, running_preds
        , average = "macro"
    )
    print("{:<5} - loss:{:.4f}, f1:{:.4f}".format(
        "val", 
        val_loss, val_f1
    ))