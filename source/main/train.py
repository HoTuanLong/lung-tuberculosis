import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from engines import *

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../../dataset/CXR-TB/train/", 
            augment = True, 
        ), 
        batch_size = 56, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../../dataset/CXR-TB/val/", 
            augment = False, 
        ), 
        batch_size = 56, 
        shuffle = False, 
    ), 
}
model = torchvision.models.swin_t(
    pretrained = True, 
)
model.head = nn.Linear(
    model.head.in_features, 2, 
)
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-5, 
)

wandb.init(
    entity = "longht", project = "CXR-TB", 
    name = "swin_t", 
)
save_ckp_dir = "../../ckps/CXR-TB/swin_t"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 50, 
    model = model, 
    optimizer = optimizer, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
)