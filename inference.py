#%%
import torch
import torch as th
from pathlib import Path
import torch.nn.functional as F
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
from transforms import train_transform
from utils import load_model, save_model, great_circle_distance, seed_everything

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'mobilenetv2_benchmark'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
THE_SEED = 42

seed_everything(THE_SEED)
dataset = Image2GeoDataset()

MODEL_PATH ='model_checkpoints/model_mobilenetv2_benchmark.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel()
load_model(model, str(MODEL_PATH))
model.to(device);

#%%
import random
dict_ = dataset[random.randint(0, len(dataset))]
imgs, labels = dict_['images'], dict_['labels']
with th.no_grad():
    img = train_transform(imgs[0])
    preds = model(img.unsqueeze(0).to(device), device)
    print(preds)
    print(labels)

imgs[0].show()
# %%
