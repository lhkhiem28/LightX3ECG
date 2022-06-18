
from imports import *

def train_fn(
    config, 
    loaders, model, 
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
        running_loss = 0.0
        running_f1 = 0.0
        for ecgs, masks in tqdm.tqdm(loaders["train"], disable = not training_verbose):
            ecgs, masks = ecgs.cuda(), masks.cuda()

            logits = model(ecgs)
            loss = F.cross_entropy(logits, masks)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss += loss.item()*ecgs.size(0)
            masks, preds = torch.argmax(masks, dim = 1), torch.argmax(torch.softmax(logits, dim = 1), dim = 1)
            running_f1 += f1_score(
                masks.cpu().numpy().flatten(), preds.detach().cpu().numpy().flatten()
                , average = "macro"
            )

        if scheduler is not None:
            scheduler.step()

        epoch_loss, epoch_f1 = running_loss/len(loaders["train"].dataset), running_f1/len(loaders["train"])
        history["train"]["loss"].append(epoch_loss), history["train"]["f1"].append(epoch_f1)
        if training_verbose:
            print("{:<5} - loss: {:.4f} - f1: {:.4f}".format(
                "train", 
                epoch_loss, epoch_f1
            ))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_f1 = 0.0
            for ecgs, masks in tqdm.tqdm(loaders["val"], disable = not training_verbose):
                ecgs, masks = ecgs.cuda(), masks.cuda()

                logits = model(ecgs)
                loss = F.cross_entropy(logits, masks)

                running_loss += loss.item()*ecgs.size(0)
                masks, preds = torch.argmax(masks, dim = 1), torch.argmax(torch.softmax(logits, dim = 1), dim = 1)
                running_f1 += f1_score(
                    masks.cpu().numpy().flatten(), preds.detach().cpu().numpy().flatten()
                    , average = "macro"
                )

        epoch_loss, epoch_f1 = running_loss/len(loaders["val"].dataset), running_f1/len(loaders["val"])
        history["val"]["loss"].append(epoch_loss), history["val"]["f1"].append(epoch_f1)
        if training_verbose:
            print("{:<5} - loss: {:.4f} - f1: {:.4f}".format(
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
        running_masks, running_preds = [], []
        for ecgs, masks in tqdm.tqdm(loaders["val"], disable = not training_verbose):
            ecgs, masks = ecgs.cuda(), masks.cuda()

            logits = model(ecgs)
            loss = F.cross_entropy(logits, masks)

            masks, preds = torch.argmax(masks, dim = 1), torch.argmax(torch.softmax(logits, dim = 1), dim = 1)
            masks, preds = masks.cpu().numpy().flatten(), preds.detach().cpu().numpy().flatten()
            running_masks.extend(masks), running_preds.extend(preds)

    print("classification-report:")
    _, optimal_thresholds = classification_report(
        logger, config.is_multilabel, 
        running_masks, running_preds
    )
    history["optimal_thresholds"] = optimal_thresholds
    np.save("{}/history.npy".format(save_ckp_path), history)

    logger.close(), print("\nFinish !!!\n")