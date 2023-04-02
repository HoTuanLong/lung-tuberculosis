import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from engines import *

test_loader = torch.utils.data.DataLoader(
    ImageDataset(
        data_dir = "../../dataset/CXR-TB/val/", 
    ), 
    batch_size = 56, 
)
model = torch.load(
    "../../ckps/CXR-TB/swin_t/best.ptl", 
    map_location = "cpu", 
)
test_fn(
    test_loader, 
    model, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
)