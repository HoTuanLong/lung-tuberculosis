import os, sys
from libs import *
from utils import dcm2image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        image_size = 512, 
        augment = False, 
    ):
        self.image_files = glob.glob(data_dir + "*/*")
        if augment:
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(height = image_size, width = image_size, ), 
                    A.HorizontalFlip(p = 0.5), 
                    A.Normalize(mean = (0.491372549, 0.482352941, 0.446666667, ), std = (0.247058824, 0.243529412, 0.261568627, ), ), AT.ToTensorV2(), 
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height = image_size, width = image_size, ), 
                    A.Normalize(mean = (0.491372549, 0.482352941, 0.446666667, ), std = (0.247058824, 0.243529412, 0.261568627, ), ), AT.ToTensorV2(), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file = self.image_files[index]
        image = dcm2image(image_file)
        # print(image.shape)

        image, label = self.transform(image = image)["image"], int(image_file.split("/")[-2])

        return image, label