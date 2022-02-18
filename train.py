#%%
import time
import torch
import random
import torch as th
from pathlib import Path
import torch.nn.functional as F
from clearml import Task, Logger
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
from transforms import train_transform
from utils import load_model, save_model, great_circle_distance, seed_everything

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'mobilenetv2_benchmark'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA = 20*60 #in seconds
TRAIN_DATA_FRACTION = 0.85
THE_SEED = 42
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1

seed_everything(THE_SEED)
task = Task.init(project_name="image2geolocation", task_name="benchmark_dg")
logger = Logger.current_logger()

#%%
dataset = Image2GeoDataset(transform=train_transform)
lentrain = int(TRAIN_DATA_FRACTION*len(dataset))
train_dataset, valid_dataset = th.utils.data.random_split(
    dataset, 
    [lentrain, len(dataset)-lentrain], 
    generator=torch.Generator().manual_seed(THE_SEED)
    )

train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=TRAIN_BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=False, 
)
valid_dataloader = DataLoader(
                    valid_dataset, 
                    batch_size=VALID_BATCH_SIZE, 
                    shuffle=False, 
                    num_workers=0, 
                    pin_memory=False, 
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel()
# load_model(model, str(MODEL_PATH))
model.to(device)
#%%%
opt = th.optim.Adam([
    {'params':model.parameters(), 'lr':2e-3},
])
# load_model(opt, str(OPTIMIZER_PATH))
# %%
step = 0
t_last_save = time.time()
for ep in range(10):
    for ibatch, train_dict in enumerate(train_dataloader):
        imgs, labels = train_dict['images'], train_dict['labels']
        imgs = imgs[random.randint(0, 3)].to(device)
        preds = model(imgs, device)

        labels = labels.to(device)
        loss = ((labels-preds)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.report_scalar("loss", "train", iteration=step , value=loss.item())

        if ibatch%10 == 0:
            print(ep, ibatch, loss.item())
        if ibatch % 10 == 0:
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    imgs, labels = valid_dict['images'], valid_dict['labels']
                    imgs = imgs[random.randint(0, 3)].to(device)
                    preds = model(imgs, device)
                    labels = labels.to(device)
                    loss = ((labels-preds)**2).mean()
                    logger.report_scalar("loss", "valid", iteration=step , value=loss.item())
                break

        if time.time() - t_last_save > SAVE_DELTA:
            save_model(model, str(MODEL_PATH))
            save_model(opt, str(OPTIMIZER_PATH))

        print(ep, step, loss.item())

        step += 1

# %%
