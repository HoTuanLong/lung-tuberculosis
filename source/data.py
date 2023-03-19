import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        image_size = 224, 
        augment = False, 
    ):
        self.image_files = glob.glob(data_dir + "*/*")
        if augment:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandAugment(
                        num_ops = 3, 
                    ), 
                    torchvision.transforms.Resize((image_size, image_size, )), 
                    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406, ], std = [0.229, 0.224, 0.225, ], ), 
                ]
            )
        else:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((image_size, image_size, )), 
                    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406, ], std = [0.229, 0.224, 0.225, ], ), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file = self.image_files[index]
        image = Image.open(image_file)

        image, label = self.transform(image), int(image_file.split("/")[-2])

        return image, label