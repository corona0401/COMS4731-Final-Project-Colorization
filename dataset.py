import os

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
        img_name = image_list[idx]
        image = io.imread(img_name)
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))
        # Convert to grayscale
        label = image.convert('LA')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample