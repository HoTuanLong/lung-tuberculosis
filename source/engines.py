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