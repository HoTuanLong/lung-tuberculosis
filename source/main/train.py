import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from engines import *

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/RLDI/train/", 
            augment = True, 
        ), 
        batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/RLDI/val/", 
            augment = False, 
        ), 
        batch_size = 32, 
        shuffle = False, 
    ), 
}
model = torchvision.models.vgg16(pretrained = True)
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features, 5, 
)
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-5, 
)

wandb.init(
    entity = "khiemlhfx", project = "RLDI", 
    name = "vgg16-imagenet", 
)
save_ckp_dir = "../../ckps/RLDI/vgg16-imagenet"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 50, 
    model = model, 
    optimizer = optimizer, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
)