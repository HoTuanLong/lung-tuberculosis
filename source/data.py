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
            self.transform = A.Compose(
                [
                    A.Resize(height = image_size, width = image_size, ), 
                    A.HorizontalFlip(), 
                    A.Affine(
                        scale = 0.4, translate_percent = 0.4, 
                    ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height = image_size, width = image_size, ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file = self.image_files[index]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2RGB, 
        )
        image, label = self.transform(image = image)["image"], int(image_file.split("/")[-2])

        return image, label