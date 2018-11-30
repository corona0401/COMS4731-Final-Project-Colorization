import os
from torch.utils.data import Dataset
import torchvision.models as models
from skimage import io
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
import torch

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
        self.vgg16 = models.vgg16(pretrained=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        rgb_img = io.imread(img_name)
        rgb_img = np.array(rgb_img, dtype=float)
        rgb_img = 1.0/255*rgb_img
        grayscale_img = gray2rgb(rgb2gray(rgb_img))
        grayscale_img = resize(grayscale_img, (240, 240, 3), mode='constant')
        grayscale_img = torch.from_numpy(grayscale_img).float()
        grayscale_img = grayscale_img.transpose(0,2)
        grayscale_img = grayscale_img.unsqueeze(0)
        with torch.no_grad():
            emd = self.vgg16(grayscale_img)
        emd = emd.numpy()
        print("embedding shape", emd.shape)
        lab_img = rgb2lab(rgb_img)
        X = lab_img[:,:,0]
        X = np.expand_dims(X, axis = 2)
        X = np.swapaxes(X, 0, 2)
        print("img shape", X.shape)
        Y = lab_img[:,:,1:] / 128
        Y = np.swapaxes(Y, 0, 2)
        print("label shape", Y.shape)
        sample = {'image': X, 'embedding': emd, 'label': Y}

        return sample


