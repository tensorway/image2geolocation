#%%
import torch
import torch as th
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transforms import train_transform, val_transform
from utils import load_model, save_model, great_circle_distance, seed_everything, draw_prediction

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'mobilenetv2_benchmark'
MODEL_NAME = 'resnet50_benchmark'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
THE_SEED = 42
TRAIN_DATA_FRACTION = 0.85

seed_everything(THE_SEED)
fill_dataset = Image2GeoDataset(transform=val_transform)
lentrain = int(TRAIN_DATA_FRACTION*len(fill_dataset))
train_fill_dataset, _ = th.utils.data.random_split(
    fill_dataset, 
    [lentrain, len(fill_dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )
train_fill_dataloader = DataLoader(
                    train_fill_dataset,
                    batch_size=32, 
                    shuffle=False, 
                    num_workers=12, 
                    pin_memory=True, 
)

seed_everything(THE_SEED)
dataset = Image2GeoDataset()
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))
train_dataset, valid_dataset = th.utils.data.random_split(
    dataset, 
    [lentrain, len(dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel('resnet50')
load_model(model, str(MODEL_PATH))
model.to(device)
model.eval()

#%%
base_labels = []
base_embeddings = []
with th.no_grad():
    model.eval()
    for train_dict in tqdm(train_fill_dataloader):
        imgs, labels = train_dict['images'], train_dict['labels']
        input = imgs[0].to(device)
        embeddings = model.embed(input)
        base_labels.append(labels)
        base_embeddings.append(embeddings)
base_labels = th.cat(base_labels, dim=0).to(device)
base_embeddings = th.cat(base_embeddings, dim=0)






#%%
# mobilenetv2 = 75
import random
temperature = 10
dict_ = valid_dataset[random.randint(0, len(valid_dataset))]
imgs, labels = dict_['images'], dict_['labels']
with th.no_grad():
    batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
    batch = th.cat(batch, dim=0).to(device)
    input = batch[0].unsqueeze(0)
    preds = model.embed(input)
    print(labels)
    preds = preds
    scores = (base_embeddings@preds.transpose(0, 1)).to(device)
    preds = base_labels * th.softmax(scores / temperature, dim=0)
    preds = preds.sum(dim=0)
    print(preds)
    dist = 0
    print(great_circle_distance(preds, labels))
    dist += great_circle_distance(preds, labels)/len(preds)
    row = preds[0]
    print("avrg dist=", dist)
    print("-"*15)

# imgs[0].show()
img = draw_prediction(preds[0], preds[1], color=(255, 0, 0))
img = draw_prediction(labels[0], labels[1], croatia_map=img, color=(0, 0, 0))
plt.imshow(img)
imgs[0]
# %%
from tqdm import tqdm
curr_dataset = valid_dataset
loop = tqdm(range(len(curr_dataset)))
base_labels = base_labels.to(device)
temperature = 10
totdist = 0 
for idx in loop:
    dict_ = curr_dataset[idx]
    imgs, labels = dict_['images'], dict_['labels']
    with th.no_grad():
        batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
        input = batch[0].to(device)
        preds = model.embed(input)
        # scores = (base_embeddings@preds.transpose(0, 1)).to(device)
        # preds = base_labels * th.softmax(scores / temperature, dim=0)
        # preds = preds.sum(dim=0)
        topsi = th.topk(scores[:, 0], k=100).indices.to(device)
        preds = base_labels[topsi].mean(dim=0)

        totdist += great_circle_distance(preds, labels)
    loop.set_description(f"{totdist/(idx+1): 4.6f}")


# %%
