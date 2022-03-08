#%%
import time
from importlib_metadata import requires
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
MODEL_NAME = 'resnet50_benchmark'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA_ALL = 20*60 #in seconds, the model that is stored and overwritten to save space
SAVE_DELTA_REVERT = 60*60 #in seconds, checkpoint models saved rarely to save storage
TRAIN_DATA_FRACTION = 0.85
THE_SEED = 42
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1


seed_everything(THE_SEED)
task = Task.init(project_name="image2geolocation", task_name="resnet50_benchmark_dg")
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
model = BenchmarkModel(model_name = 'resnet50', NUM_OF_CLASSES=dataset.number_of_classes())
load_model(model, str(MODEL_PATH))
model.to(device)
#%%%
opt = th.optim.Adam([
    {'params':model.parameters(), 'lr':1e-4},
])
load_model(opt, str(OPTIMIZER_PATH))
# %%
step = 0
t_last_save_revert = time.time()
t_last_save_all = time.time()
for ep in range(4*20):
    for ibatch, train_dict in enumerate(train_dataloader):
        imgs, labels = train_dict['images'], train_dict['labels']
        
        #imgs = imgs[random.randint(0, 3)].to(device)
        batch = th.cat([img for img in imgs], dim=0)
        batch = batch.to(device)
        labels = labels.to(device)
        
        #print("batch.size(): ", batch.size())
        preds = model(batch, device)
        #print(preds)
        
        tuple_of_y = []
        #accept only third label
        #then one-hot
        for i in range(TRAIN_BATCH_SIZE):
            x = th.tensor(int(labels[i][2]))
            l = th.tensor([[x], [x], [x], [x]])
            y = th.zeros([4, dataset.number_of_classes()])
            y = y.scatter_(1, l, 1)
            y = th.unsqueeze(y,0)
            
            tuple_of_y.append(y)
        #print(y_onehot.size())
        #print(preds.size())
        y_onehot = th.cat(tuple_of_y)
        y_onehot = th.reshape(y_onehot, (1, 4*TRAIN_BATCH_SIZE,dataset.number_of_classes()))
        
        #print(preds)
        #print(y_onehot)
        #loss = F.binary_cross_entropy(y_onehot, preds)
        #print(loss)


        loss = th.nn.CrossEntropyLoss()
        input = preds
        target = y_onehot
        #print(input.size())
        #print(target.size())
        output = loss(input, target)


        opt.zero_grad()
        output.backward()
        opt.step()


        if ibatch%10 == 0:
            logger.report_scalar("loss", "train", iteration=step , value=output.item())
            print(ep/4, step, output.item())
            #print(y_onehot)
            #print(preds)
        if ibatch % 150 == 0:
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    imgs, labels = valid_dict['images'], valid_dict['labels']

                    batch = th.cat([img for img in imgs], dim=0)
                    batch = batch.to(device)
                    labels = labels.to(device)

                    preds = model(batch, device)

                    tuple_of_y = []
                    #accept only third label
                    #then one-hot
                    for i in range(VALID_BATCH_SIZE):
                        x = th.tensor(int(labels[i][2]))
                        l = th.tensor([[x], [x], [x], [x]])
                        y = th.zeros([4, dataset.number_of_classes()])
                        y = y.scatter_(1, l, 1)
                        y = th.unsqueeze(y,0)
                        
                        tuple_of_y.append(y)
                    #print(y_onehot.size())
                    #print(preds.size())
                    y_onehot = th.cat(tuple_of_y)
                    y_onehot = th.reshape(y_onehot, (1,4*VALID_BATCH_SIZE,dataset.number_of_classes()))

                    #print(y_onehot.size())
                    #print(preds.size())

                    #loss = F.binary_cross_entropy(y_onehot, preds)
                    
                    loss = th.nn.CrossEntropyLoss()
                    input = preds
                    target = y_onehot
                    output = loss(input, target)

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
# %%
