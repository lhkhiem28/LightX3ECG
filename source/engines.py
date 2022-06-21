
import os, sys
from imports import *

def train_fn(
    config, 
    loaders, model, 
    regressor_lambda, 
    num_epochs, 
    optimizer, 
    scheduler = None, 
    save_ckp_path = None, training_verbose = True, 
):
    history = {
        "train": {"loss": [], "f1": []}, 
        "val": {"loss": [], "f1": []}, 
    }
    logger = open("{}/log.txt".format(save_ckp_path), "w")
    logger.write("\n{}".format(model.name))
    logger.write("\nNumber of parameters of {}: {}".format(model.name, get_number_parameters(model)))

    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = config.device_ids)

    best_f1 = 0.0
    for epoch in tqdm.tqdm(range(1, num_epochs + 1), disable = training_verbose):
        if training_verbose:
            print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + "-"*16)

        model.train()
        running_loss, running_sub_loss = 0.0, 0.0
        running_labels, running_preds = [], []
        for (ecgs, demographics), r_counts, labels in tqdm.tqdm(loaders["train"], disable = not training_verbose):
            (ecgs, demographics), r_counts, labels = (ecgs.cuda(), demographics.cuda()), r_counts.cuda(), labels.cuda()

            logits, sub_logits = model((ecgs, demographics))
            loss, sub_loss = F.cross_entropy(logits, labels) if not config.is_multilabel else F.binary_cross_entropy_with_logits(logits, labels), F.l1_loss(sub_logits, r_counts)

            (loss + regressor_lambda*sub_loss).backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_sub_loss = running_loss + loss.item()*ecgs.size(0), running_sub_loss + sub_loss.item()*ecgs.size(0)
            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config.is_multilabel else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1, 0))
            running_labels.extend(labels), running_preds.extend(preds)

        if (scheduler is not None) and (not epoch > scheduler.T_max):
            scheduler.step()

        (epoch_loss, epoch_sub_loss), epoch_f1 = (running_loss/len(loaders["train"].dataset), running_sub_loss/len(loaders["train"].dataset)), f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        history["train"]["loss"].append(epoch_loss), history["train"]["f1"].append(epoch_f1)
        if training_verbose:
            print("{:<5} - *(loss: {:.4f}, sub_loss: {:.4f}), f1: {:.4f}".format(
                "train", 
                *(epoch_loss, epoch_sub_loss), epoch_f1
            ))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_labels, running_preds = [], []
            for (ecgs, demographics), r_counts, labels in tqdm.tqdm(loaders["val"], disable = not training_verbose):
                (ecgs, demographics), r_counts, labels = (ecgs.cuda(), demographics.cuda()), r_counts.cuda(), labels.cuda()

                logits, sub_logits = model((ecgs, demographics))
                loss = F.cross_entropy(logits, labels) if not config.is_multilabel else F.binary_cross_entropy_with_logits(logits, labels)

                running_loss += loss.item()*ecgs.size(0)
                labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config.is_multilabel else list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1, 0))
                running_labels.extend(labels), running_preds.extend(preds)

        epoch_loss, epoch_f1 = running_loss/len(loaders["val"].dataset), f1_score(
            running_labels, running_preds
            , average = "macro"
        )
        history["val"]["loss"].append(epoch_loss), history["val"]["f1"].append(epoch_f1)
        if training_verbose:
            print("{:<5} - loss: {:.4f}, f1: {:.4f}".format(
                "val", 
                epoch_loss, epoch_f1
            ))

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.module, "{}/best.ptl".format(save_ckp_path))

    print("\nValidation ...\n" + " = "*16)
    model = torch.load("{}/best.ptl".format(save_ckp_path), map_location = "cuda")
    model = nn.DataParallel(model, device_ids = config.device_ids)

    with torch.no_grad():
        model.eval()
        running_labels, running_preds = [], []
        for (ecgs, demographics), r_counts, labels in tqdm.tqdm(loaders["val"], disable = not training_verbose):
            (ecgs, demographics), r_counts, labels = (ecgs.cuda(), demographics.cuda()), r_counts.cuda(), labels.cuda()

            logits, sub_logits = model((ecgs, demographics))
            loss = F.cross_entropy(logits, labels) if not config.is_multilabel else F.binary_cross_entropy_with_logits(logits, labels)

            labels, preds = list(labels.data.cpu().numpy()), list(torch.max(logits, 1)[1].detach().cpu().numpy()) if not config.is_multilabel else list(torch.sigmoid(logits).detach().cpu().numpy())
            running_labels.extend(labels), running_preds.extend(preds)

    print("classification-report:")
    _, optimal_thresholds = classification_report(
        logger, config.is_multilabel, 
        running_labels, running_preds
    )
    history["optimal_thresholds"] = optimal_thresholds
    np.save("{}/history.npy".format(save_ckp_path), history)

    logger.close(), print("\nFinish !!!\n")