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
from utils import load_model, save_model, great_circle_distance, seed_everything, draw_prediction

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'mobilenetv2_benchmark2'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
THE_SEED = 42
TRAIN_DATA_FRACTION = 0.85

seed_everything(THE_SEED)
dataset = Image2GeoDataset()
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))
train_dataset, valid_dataset = th.utils.data.random_split(
    dataset, 
    [lentrain, len(dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )

MODEL_PATH ='model_checkpoints/model_mobilenetv2_benchmark.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel()
load_model(model, str(MODEL_PATH))
model.to(device)

#%%
import random
dict_ = valid_dataset[random.randint(0, len(valid_dataset))]
imgs, labels = dict_['images'], dict_['labels']
with th.no_grad():
    batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
    batch = th.cat(batch, dim=0).to(device)
    preds = model(batch, device)
    print(preds)
    print(labels)
    dist = 0
    for row in preds:
        print(great_circle_distance(row, labels))
        dist += great_circle_distance(row, labels)/len(preds)
    row = preds[0]
    print("avrg dist=", dist)
    print("-"*15)

imgs[0].show()
img = draw_prediction(row[0], row[1], color=(255, 0, 0))
img = draw_prediction(labels[0], labels[1], croatia_map=img, color=(0, 0, 0))
plt.imshow(img)
# %%
from tqdm import tqdm
curr_dataset = valid_dataset
loop = tqdm(range(len(curr_dataset)))
totdist = 0 
for idx in loop:
    dict_ = curr_dataset[random.randint(0, len(curr_dataset))]
    imgs, labels = dict_['images'], dict_['labels']
    with th.no_grad():
        batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
        batch = th.cat(batch, dim=0).to(device)
        preds = model(batch, device)
        dist = 0
        for row in preds:
            dist += great_circle_distance(row, labels)/len(preds)
        print(great_circle_distance(preds[0], labels))
    totdist += dist
    loop.set_description(f"{totdist/(idx+1): 4.6f}")

#%%
loop = tqdm(range(len(train_dataset)))
totdist_la = 0 
totdist_lo = 0 
for idx in loop:
    dict_ = train_dataset[idx]
    imgs, labels = dict_['images'], dict_['labels']
    totdist_la += labels[0]
    totdist_lo += labels[1]
    loop.set_description(f"{totdist_la/(idx+1): 4.6f} {totdist_lo/(idx+1): 4.6f}")

#%%
preds = [45.124367, 16.389811]
loop = tqdm(range(len(valid_dataset)))
totdist = 0 
for idx in loop:
    dict_ = valid_dataset[idx]
    imgs, labels = dict_['images'], dict_['labels']
    totdist += great_circle_distance(preds, labels)
    loop.set_description(f"{totdist/(idx+1): 4.6f}")

