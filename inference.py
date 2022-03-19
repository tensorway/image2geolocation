#%%
import torch
import random
import torch as th
from tqdm import tqdm
from pathlib import Path
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transforms import train_transform, val_transform
from utils import load_model, great_circle_distance, seed_everything, draw_prediction
from ranking import create_base_embeddings_and_labels, eval_ranking_loop


MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'mobilenetv2_benchmark'
MODEL_NAME = 'mobilenet_ranking'
MODEL_NAME = 'mobilenet_ranking'
MODEL_NAME = 'resnet50_ranking'
MODEL_NAME = 'simclrl'

MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
THE_SEED = 42
VALID_BATCH_SIZE = 64
TRAIN_DATA_FRACTION = 0.85
TRAIN_DATABASE_BATCH_SIZE = 64

seed_everything(THE_SEED)


def contrastive_collate_fn (batch):
    toret = []
    for el_list in batch:
        for el in el_list:
            toret.append(el.unsqueeze(0))
    return th.cat(toret, dim=0)

dataset = Image2GeoDataset(transform=train_transform)
idxs = list(range(len(dataset)))
random.shuffle(idxs)
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))

train_database_dataset = Image2GeoDataset(transform=val_transform, csv_which_idxs2use=idxs[:lentrain])
valid_dataset          = Image2GeoDataset(transform=val_transform, csv_which_idxs2use=idxs[lentrain:])

train_database_dataloader = DataLoader(
                    train_database_dataset, 
                    batch_size=TRAIN_DATABASE_BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=6, 
                    pin_memory=True, 
)
valid_dataloader = DataLoader(
                    valid_dataset, 
                    batch_size=VALID_BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=6, 
                    pin_memory=True, 
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel('resnet50')
load_model(model, str(MODEL_PATH))
model.to(device)
model.eval()

#%%
base_embeddings, base_labels = create_base_embeddings_and_labels(model, train_database_dataloader, device)
#%%

average_great_circle_distance = eval_ranking_loop(model, valid_dataset, device, base_embeddings, base_labels, 0.001)
print(average_great_circle_distance)

#%%
# # # mobilenetv2 = 75

# temperature = 0.001
# sample = valid_dataset[random.randint(0, len(valid_dataset))]
# imgs, labels = sample['images'], sample['labels']
# with th.no_grad():
#     batch = tuple(img.unsqueeze(0) for img in imgs)
#     batch = th.cat(batch, dim=0).to(device)
#     input = batch[0].unsqueeze(0)
#     preds = model.embed(input)
#     print(labels)
#     preds = preds
#     scores = (base_embeddings@preds.transpose(0, 1)).to(device)
#     preds = base_labels * th.softmax(scores / temperature, dim=0)
#     preds = preds.sum(dim=0)
#     print(preds)
#     dist = 0
#     print(great_circle_distance(preds, labels))
#     dist += great_circle_distance(preds, labels)/len(preds)
#     row = preds[0]
#     print("avrg dist=", dist)
#     print("-"*15)

# # imgs[0].show()
# img = draw_prediction(preds[0], preds[1], color=(255, 0, 0))
# img = draw_prediction(labels[0], labels[1], croatia_map=img, color=(0, 0, 0))
# plt.imshow(img)
# imgs[0]

# # %%
# from tqdm import tqdm
# curr_dataset = valid_dataset
# base_labels = base_labels.to(device)



# eval_ranking_loop(model, eval)

# # %%
