#%%
import torch
import torch as th
from pathlib import Path
import torch.nn.functional as F
from models import BenchmarkModel
from dataset import ClassificationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transforms import train_transform, val_transform
from utils import load_model, save_model, great_circle_distance, seed_everything, draw_prediction

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
#MODEL_NAME = 'mobilenetv_benc'
#MODEL_NAME = 'resnet50_bench'
#MODEL_NAME = 'resnet50_bench8923'
#MODEL_NAME = 'mobilenet_4classes'
MODEL_NAME = 'mobilenet_4classes_1'

MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_classification_'+MODEL_NAME+'.pt')
THE_SEED = 42
TRAIN_DATA_FRACTION = 0.85

seed_everything(THE_SEED)
dataset = ClassificationDataset()
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))
train_dataset, valid_dataset2 = th.utils.data.random_split(
    dataset, 
    [lentrain, len(dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel(model_name='mobilenet_v2', NUM_OF_CLASSES = dataset.number_of_classes())
load_model(model, str(MODEL_PATH))
model.to(device)

#%%
import random
dict_ = valid_dataset2[random.randint(0, len(valid_dataset2))]
imgs, labels = dict_['images'], dict_['labels']
with th.no_grad():
    batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
    batch = th.cat(batch, dim=0).to(device)
    preds = model(batch, device)
    #print(preds)
    
    print(labels)
    
    guess = th.max(th.squeeze(preds,0), 1)
    print(guess.indices)

    centroid = (0.0,0.0)

    original_labels = labels[0], labels[1]
    la, lo = 0, 0
    for i in range(4):
        temp = dataset.centroid(int(guess.indices[i]))
        la += temp[0]/4
        lo += temp[1]/4
    centroid = (la,lo)
    print(great_circle_distance(centroid, original_labels))

    #print(th.mode(guess,0))
    

# %%
# resnet50 = 62.98
# mobilenetv2 = 98
from tqdm import tqdm
curr_dataset = valid_dataset2
loop = tqdm(range(len(curr_dataset)))
totdist = 0 
model.eval()
for idx in loop:
    dict_ = curr_dataset[random.randint(0, len(curr_dataset))]
    imgs, labels = dict_['images'], dict_['labels']
    with th.no_grad():
        batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
        batch = th.cat(batch, dim=0).to(device)
        preds = model(batch, device)
        dist = 0
        
        guess = th.max(th.squeeze(preds,0), 1)
        la, lo = 0, 0
        original_labels = (labels[0], labels[1])
        for i in range(4):
            temp = dataset.centroid(int(guess.indices[i]))
            la += temp[0]/4
            lo += temp[1]/4


        centroid = (la,lo)
        dist += great_circle_distance(centroid, original_labels)

        #print(great_circle_distance(preds[0], labels))
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


# %%
