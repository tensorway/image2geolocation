#%%
import time
import torch
import random
import torch as th
from pathlib import Path
import torch.nn.functional as F
from clearml import Task, Logger
from models import BenchmarkModel
from dataset import ClassificationDataset
from torch.utils.data import DataLoader
from transforms import train_transform
from utils import load_model, save_model, great_circle_distance, seed_everything

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'classification_resnet50_bench'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA_ALL = 10*60 #in seconds, the model that is stored and overwritten to save space
SAVE_DELTA_REVERT = 20*60 #in seconds, checkpoint models saved rarely to save storage
TRAIN_DATA_FRACTION = 0.85
THE_SEED = 42
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32

seed_everything(THE_SEED)
task = Task.init(project_name="image2geolocation", task_name="classification_bench_resnet")
logger = Logger.current_logger()

#%%
dataset = ClassificationDataset(transform=train_transform)
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
                    num_workers=10, 
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
model = BenchmarkModel(model_name='resnet50')
# load_model(model, str(MODEL_PATH))
model.to(device)
#%%%
opt = th.optim.Adam([
    {'params':model.model.parameters(), 'lr':3e-5},
    {'params':model.classifier.parameters(), 'lr':3e-3},
])
# load_model(opt, str(OPTIMIZER_PATH))
# %%
step = 0
t_last_save_revert = time.time()
t_last_save_all = time.time()
loss = th.nn.CrossEntropyLoss()

for ep in range(4*30):
    for ibatch, train_dict in enumerate(train_dataloader):
        imgs, labels = train_dict['images'], train_dict['labels']
        
        batch = imgs[random.randint(0, 3)]
        batch = batch.to(device)
        labels = labels.to(device)
        
        preds = model(batch, device)
        tuple_of_y = []

        for i in range(len(batch)):
            x = th.tensor(int(labels[i][2]))
            y = th.tensor([x])
            tuple_of_y.append(y)


        target = th.cat(tuple_of_y, dim=0).to(device)
        output = loss(preds, target)



        opt.zero_grad()
        output.backward()
        opt.step()

        if ibatch%10 == 0:
            logger.report_scalar("loss", "train", iteration=step , value=output.item())
            print(ep/4, step, output.item())
        if ibatch % 150 == 0:
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    imgs, labels = valid_dict['images'], valid_dict['labels']

                    batch = imgs[random.randint(0, 3)]

                    batch = batch.to(device)
                    labels = labels.to(device)

                    preds = model(batch, device)

                    tuple_of_y = []
                    for i in range(len(batch)):
                        x = th.tensor(int(labels[i][2]))
                        y = th.tensor([x])
                        tuple_of_y.append(y)


                    target = th.cat(tuple_of_y, dim=0).to(device)
                    output = loss(preds, target)

                    logger.report_scalar("loss", "valid", iteration=step , value=output.item())
                break

        if time.time() - t_last_save_all > SAVE_DELTA_ALL:
            save_model(model, str(MODEL_PATH))
            save_model(opt, str(OPTIMIZER_PATH))
            t_last_save_all = time.time()

        if time.time() - t_last_save_revert > SAVE_DELTA_REVERT:
            save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
            save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
            t_last_save_revert = time.time()
        

        step += 1



# %%
save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
# %%
save_model(model, str(MODEL_PATH))
save_model(opt, str(OPTIMIZER_PATH))
#%%
2+2
# %%
loss = th.nn.CrossEntropyLoss()
input = torch.randn(21, 21)
# input = input / input.sum(dim=-1, keepdim=True)
input = th.ones(21, 21)*0.2/21 + th.eye(21)*0.8
target = th.range(0, 20, dtype=th.long)
output = loss(input, target)
output
# %%
