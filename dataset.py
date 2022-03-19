#%%
import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch as th
from tqdm import tqdm
import random
import pickle
from utils import great_circle_distance
import numpy as np
from ranking import transform0, transform90, transform180, transform270
class Image2GeoDataset(Dataset):
    """Image2GeoDataset dataset."""

    def __init__(
        self, 
        images_folder='dataset/data', 
        csv_file='dataset/data.csv', 
        transform=None,
        csv_which_idxs2use=None
        ):
        """
        Args:
            images_folder (string): Directory with directories of images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        if csv_which_idxs2use:
            uuid, latitude, longitude = [], [], []
            for idx in csv_which_idxs2use:
                uuid.append(self.csv['uuid'][idx])
                latitude.append(self.csv['latitude'][idx])
                longitude.append(self.csv['longitude'][idx])
            self.csv = {'uuid':uuid, 'latitude':latitude, 'longitude':longitude}
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.csv['uuid'])

    def get_labels(self):
        latitudes = self.csv['latitude']
        longitudes = self.csv['longitude']
        return [ (la, lo) for la, lo in zip(latitudes, longitudes) ]

    def __getitem__(self, idx, differentiate_views=True):
        images = []
        pattern = os.path.join(self.images_folder, self.csv['uuid'][idx], '*')
        for file_name in glob.glob(pattern):
            image = Image.open(file_name)
            if self.transform:
                image = self.transform(image)
                if differentiate_views:
                    if '270' in file_name:
                        transform270(image)
                    elif '180' in file_name:
                        transform180(image)
                    elif '90' in file_name:
                        transform90(image)
                    else:
                        transform0(image)
            images.append(image)

        labels = [self.csv['latitude'][idx], self.csv['longitude'][idx]]
        labels = th.tensor(labels, dtype=th.float32)
        sample = {'images': images, 'labels': labels}

        return sample



class ContrastiveImage2GeoDataset(Image2GeoDataset):
    def __init__(
            self, 
            images_folder='dataset/data', 
            csv_file='dataset/data.csv', 
            transform=None, 
            csv_which_idxs2use=None,
            ordering_file='ordering.pckl',
            sampling_func = lambda : np.clip((( np.random.pareto(10, 100) + 1) * 10-10).astype('int'), 0, 199)[0] #pareto https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html 
            # sampling_func = lambda: random.randint(0, 199)
        ):
        super().__init__(images_folder, csv_file, transform, csv_which_idxs2use)
        try:
            with open(ordering_file, 'rb') as f:
                self.sorted_distances = pickle.load(f)
        except:
            self.make_ordering(save_file=ordering_file)
        self.sampling_func = sampling_func


    def make_ordering(self, save_file):
        '''
        just a simple distance calculation between every element and
         every other element
        works suprisingly fast, around two min (the result is saved)
        the distances and labels are then sorted, so
        every element has a sorted list of closest other elements
        '''
        sorted_distances = []
        csv = list(zip(self.csv['latitude'], self.csv['longitude']))
        for i, el in enumerate(tqdm(csv)):
            distances = []
            for j, el2 in enumerate(csv):
                if i!=j:
                    val = great_circle_distance(el, el2)
                    distances.append((val, j))
            distances.sort()
            sorted_distances.append(distances[:200])
        with open(save_file, 'wb') as f:
            pickle.dump(sorted_distances, f)

        self.sorted_distances = sorted_distances


    def __getitem__(self, idx):
        imgs = super().__getitem__(idx, differentiate_views=True)['images']
        idx = self.sorted_distances[idx][self.sampling_func()][1]
        imgs2 = super().__getitem__(idx, differentiate_views=True)['images']
        imgs = imgs + imgs2
        return imgs

    

    

#%%




if __name__ == '__main__':
    dataset = ContrastiveImage2GeoDataset()
    idx = random.randint(0, len(dataset)-1)
    print(dataset[idx])
    dataset[idx][0]
    # %%
    import matplotlib.pyplot as plt
    import numpy as np
    a, m = 5., 10.  # shape and mode

    s = np.clip( ((np.random.pareto(a, 100) + 1) * m-m).astype('int'), a_max=100, a_min=0 )

    count, bins, _ = plt.hist(s, 100, density=1)

    plt.show()

    # %%
