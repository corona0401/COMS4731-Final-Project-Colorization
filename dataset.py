import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class PlaceDataset(Dataset):

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        imagelist = []
        for filename in os.listdir(image_dir):
            path = os.path.join(image_dir,filename)
            imagelist.append(path)
        self.image_list = imagelist
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        pil2tensor = transforms.ToTensor()
        img_name = self.image_list[idx]
        rgb_image = Image.open(img_name)
        rgb_image = pil2tensor(rgb_image)
        r_image = rgb_image[0]
        g_image = rgb_image[1]
        b_image = rgb_image[2]
        grayscale_image = (r_image + g_image + b_image).div(3.0)
        grayscale_image = grayscale_image.unsqueeze(0)
        sample = {'image': grayscale_image, 'label': rgb_image}

        return sample
