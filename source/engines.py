import os, sys
from libs import *

def train_fn(
    train_loaders, num_epochs, 
    model, 
    optimizer, 
    device = torch.device("cpu"), 
    save_ckp_dir = "./", 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        model.train()
        running_loss, running_corrects,  = 0.0, 0.0, 
        for images, labels in tqdm.tqdm(train_loaders["train"]):
            images, labels = images.to(device), labels.to(device)

            logits = model(images.float())
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
        train_loss, train_accuracy,  = running_loss/len(train_loaders["train"].dataset), running_corrects/len(train_loaders["train"].dataset), 
        wandb.log({"train_loss":train_loss, "train_accuracy":train_accuracy, }, step = epoch)
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "train", 
            train_loss, train_accuracy, 
        ))

        with torch.no_grad():
            model.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            for images, labels in tqdm.tqdm(train_loaders["val"]):
                images, labels = images.to(device), labels.to(device)

                logits = model(images.float())
                loss = F.cross_entropy(logits, labels)

                running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
        val_loss, val_accuracy,  = running_loss/len(train_loaders["val"].dataset), running_corrects/len(train_loaders["val"].dataset), 
        wandb.log({"val_loss":val_loss, "val_accuracy":val_accuracy, }, step = epoch)
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "val", 
            val_loss, val_accuracy, 
        ))
        if val_accuracy > best_accuracy:
            torch.save(model, "{}/best.ptl".format(save_ckp_dir))
            best_accuracy = val_accuracy

    print("\nFinish Training ...\n" + " = "*16)
    return {
        "train_loss":train_loss, "train_accuracy":train_accuracy, 
        "val_loss":val_loss, "val_accuracy":val_accuracy, 
    }

def test_fn(
    test_loader, 
    model, 
    device = torch.device("cpu"), 
):
    print("\nStart Testing ...\n" + " = "*16)
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        running_loss, running_corrects,  = 0.0, 0.0, 
        running_labels, running_predictions,  = [], [], 
        for images, labels in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images.float())
            loss = F.cross_entropy(logits, labels)

            running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
            running_labels, running_predictions,  = running_labels + labels.data.cpu().numpy().tolist(), running_predictions + torch.max(logits, 1)[1].detach().cpu().numpy().tolist(), 
    test_loss, test_accuracy,  = running_loss/len(test_loader.dataset), running_corrects/len(test_loader.dataset), 
    print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
        "test", 
        test_loss, test_accuracy, 
    ))
    print(metrics.classification_report(
        running_labels, running_predictions, 
        digits = 4, 
    ))

    print("\nFinish Testing ...\n" + " = "*16)
    return {
        "test_loss":test_loss, "test_accuracy":test_accuracy, 
    }