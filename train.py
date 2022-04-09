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
from itertools import chain 
from utils import load_model, save_model, great_circle_distance, seed_everything

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'nvidia_efficientnet_widese_b4_gaussian'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA_ALL = 10*60 #in seconds, the model that is stored and overwritten to save space
SAVE_DELTA_REVERT = 20*60 #in seconds, checkpoint models saved rarely to save storage
TRAIN_DATA_FRACTION = 0.85
THE_SEED = 42
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

seed_everything(THE_SEED)
task = Task.init(project_name="image2geolocation", task_name="nvidia_efficientnet_widese_b4_gaussian_finetune")
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
model = BenchmarkModel('nvidia_efficientnet_widese_b4')
load_model(model, 'model_checkpoints/model_efficientnetb4.pt')
model.to(device)
#%%%
opt = th.optim.Adam([
    # {'params':model.model.parameters(), 'lr':1e-4},
    # {'params':model.mean.parameters(), 'lr':1e-3},
    {'params':model.cov.parameters(), 'lr':1e-4},
])
for p in chain(model.model.parameters(), model.classifier.parameters()):
    p.requires_grad = False
# load_model(opt, str(OPTIMIZER_PATH))
# %%
step = 0
t_last_save_revert = time.time()
t_last_save_all = time.time()
scaler = torch.cuda.amp.GradScaler()

def multivariate_gaussian_nll_loss(target, mu, conv, device, log_eps=1e-6, add_eps=1e-6):
    '''
    eps is a parameter that serves to improve training stability
    target and mu are tensors of the shape batch_size, 2 and 
    conv is of the shape batch_size, 2, 2
    the function handles conv inverse errors as well as 
    '''
    target, mu = target.unsqueeze(-1), mu.unsqueeze(-1)
    n_failed = 0
    while True:
        try:
            inv = th.inverse(conv)
            break
        except Exception as e:
            print('falied to invert a matrix', e, n_failed, conv)
            n_failed += 1
            if n_failed < 5:
                conv = conv + th.tensor([[[add_eps, -add_eps], [add_eps, add_eps]]], device=device)
            else:
                inv = 0
                break

    loss_determinant = th.log( th.clip( th.det(conv), min=log_eps ))
    loss_mean = th.bmm(th.bmm((mu-target).transpose(2, 1), inv), mu-target)

    return loss_determinant, loss_mean

for ep in range(4*20):
    for ibatch, train_dict in enumerate(train_dataloader):
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            imgs, labels = train_dict['images'], train_dict['labels']
            imgs = imgs[random.randint(0, 3)].to(device)
            mu, conv = model(imgs, device)

            labels = labels.to(device)
            loss_determinant, loss_mean = multivariate_gaussian_nll_loss(labels, mu, conv, device)
            loss = (loss_determinant/10+loss_mean).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


        if ibatch%10 == 0:
            logger.report_scalar("loss", "train", iteration=step , value=loss.item())
            logger.report_scalar("loss_determinant", "train", iteration=step , value=loss_determinant.mean().item())
            logger.report_scalar("loss_mean", "train", iteration=step , value=loss_mean.mean().item())
            logger.report_scalar("det", "valid", iteration=step , value=th.det(conv).mean().item())
            logger.report_scalar("squared_dist", "train", iteration=step , value=((labels-mu)**2).mean().item())
            dist = 0
            for row, lrow in zip(mu, labels):
                dist += great_circle_distance(row, lrow)/len(mu)
            logger.report_scalar("great_circle_distance", "train", iteration=step , value=dist)
            print(ep/4, step, loss.item())
            
        if ibatch % 150 == 0:
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    with torch.cuda.amp.autocast():
                        imgs, labels = valid_dict['images'], valid_dict['labels']
                        imgs = imgs[random.randint(0, 3)].to(device)
                        mu, conv = model(imgs, device)
                        labels = labels.to(device)
                        loss_determinant, loss_mean = multivariate_gaussian_nll_loss(labels, mu, conv, device)
                        loss = (loss_determinant/10+loss_mean).mean()   
                        dist = 0
                        for row, lrow in zip(mu, labels):
                            dist += great_circle_distance(row, lrow)/len(mu)
                        logger.report_scalar("great_circle_distance", "valid", iteration=step , value=dist)
                        logger.report_scalar("loss", "valid", iteration=step , value=loss.item())
                        logger.report_scalar("loss_determinant", "valid", iteration=step , value=loss_determinant.mean().item())
                        logger.report_scalar("loss_mean", "valid", iteration=step , value=loss_mean.mean().item())
                        logger.report_scalar("det", "valid", iteration=step , value=th.det(conv).mean().item())
                        logger.report_scalar("squared_dist", "valid", iteration=step , value=((labels-mu)**2).mean().item())
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
