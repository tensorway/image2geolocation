#%%
import torch
import torch as th
from pathlib import Path
import torch.nn.functional as F
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transforms import train_transform, val_transform
from utils import load_model, save_model, great_circle_distance, seed_everything, draw_prediction, draw_gaussian_prediction, multiply_and_normalize_gaussians

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'efficientnetv2_rw_m_2'


MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
THE_SEED = 42
TRAIN_DATA_FRACTION = 0.85

seed_everything(THE_SEED)
dataset = Image2GeoDataset(cleaned=False)
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))
_, valid_dataset2 = th.utils.data.random_split(
    dataset, 
    [lentrain, len(dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# model = BenchmarkModel(model_name='efficientnetv2_rw_m')
# load_model(model, str(MODEL_PATH))
# model.to(device)

#%%
import random
import matplotlib.pyplot as plt
dict_ = valid_dataset2[random.randint(0, len(valid_dataset2))]
imgs, labels = dict_['images'], dict_['labels']

with th.no_grad():
    batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
    batch = th.cat(batch, dim=0).to(device)
    mus, covs = model(batch, device)
    mu, cov = multiply_and_normalize_gaussians(mus, covs)
    mu, cov = mu.to('cpu'), cov.to('cpu')

    print(mu)
    print(labels)
    print(great_circle_distance(mu, labels))

img = draw_prediction(labels[0], labels[1], croatia_map=None, color=(0, 0, 255))
img, _ = draw_gaussian_prediction(mu, cov, croatia_map=img);

plt.imshow(img)

imgs[0]
# %%
# efficientnet b4 = 38.68
# resnet152_cleaned on clean = 53.52
# resnet152 = 55.79
# resnet50 = 62.98
# mobilenetv2 = 98
from tqdm import tqdm
curr_dataset = valid_dataset2
loop = tqdm(range(len(curr_dataset)))
totdist = 0 
model.eval() 
for idx in loop:
    dict_ = curr_dataset[idx]
    imgs, labels = dict_['images'], dict_['labels']
    with th.no_grad():
        batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
        batch = th.cat(batch, dim=0).to(device)
        mus, covs = model(batch, device)
        mu, cov = multiply_and_normalize_gaussians(mus, covs)
        dist = 0 #great_circle_distance(mu, labels)
        for row in mus:
            dist += great_circle_distance(row, labels)/len(mu)
        #print(great_circle_distance(mu[0], labels))
    totdist += dist
    loop.set_description(f"{totdist/(idx+1): 4.6f}")



# %%
