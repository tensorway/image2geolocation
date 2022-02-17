import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch as th
import random

class Image2GeoDataset(Dataset):
    """Image2GeoDataset dataset."""

    def __init__(
        self, 
        images_folder='dataset/data', 
        csv_file='dataset/data.csv', 
        transform=None
        ):
        """
        Args:
            images_folder (string): Directory with directories of images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def get_labels(self):
        latitudes = self.csv['latitude']
        longitudes = self.csv['longitude']
        return [ (la, lo) for la, lo in zip(latitudes, longitudes) ]

    def __getitem__(self, idx):
        images = []
        pattern = os.path.join(self.images_folder, self.csv['uuid'][idx], '*')
        for file_name in glob.glob(pattern):
            image = Image.open(file_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        labels = [self.csv['latitude'][idx], self.csv['longitude'][idx]]
        labels = th.tensor(labels, dtype=th.float32)
        sample = {'images': images, 'labels': labels}

        return sample


if __name__ == '__main__':
    dataset = Image2GeoDataset()
    idx = random.randint(0, len(dataset)-1)
    print(dataset[idx])
    dataset[idx]['images'][0].show()