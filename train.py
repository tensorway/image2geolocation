#%%
import time
from numpy import block
import torch
import random
import torch as th
from pathlib import Path
import torch.nn.functional as F
from clearml import Task, Logger
from models import BenchmarkModel
from torch.utils.data import DataLoader
from transforms import de_train_transform, train_transform, val_transform
from dataset import ContrastiveImage2GeoDataset, Image2GeoDataset
from utils import great_circle_distance, load_model, save_model, seed_everything
from ranking import contrastive_labels_mat, create_base_embeddings_and_labels, eval_ranking_loop, get_zeroing_mat


THE_SEED = 42

SAVE_DELTA_ALL = 10*60          #in seconds, the model that is stored and overwritten to save space
VALID_TIME_DELTA = 10*60        #in seconds, how often to validate the model, validation is more expensive so less than in usual ml workloads
SAVE_DELTA_REVERT = 20*60       #in seconds, checkpoint models saved rarely to save storage

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 64
TRAIN_DATA_FRACTION = 0.85
TRAIN_DATABASE_BATCH_SIZE = 64

N_EVAL_VECTORS = 250
VALID_TEMPERATURE = 0.001
N_DATABASE_VECTORS = 2000

MODEL_NAME = 'mobilenet_ranking'
MODEL_NAME = 'resnet50_ranking'
MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')

seed_everything(THE_SEED)
task = Task.init(project_name="image2geolocation", task_name="resnet50_ranking")
logger = Logger.current_logger()





#%%
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

train_dataset = ContrastiveImage2GeoDataset(
    transform=train_transform, 
    csv_which_idxs2use=idxs[:lentrain], 
    ordering_file='ordering_files/train_seed42_085.pckl'
    )

train_database_dataset = Image2GeoDataset(transform=val_transform, csv_which_idxs2use=idxs[:N_DATABASE_VECTORS])
valid_dataset          = Image2GeoDataset(transform=val_transform, csv_which_idxs2use=idxs[lentrain:lentrain+N_EVAL_VECTORS])

train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=TRAIN_BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=8, 
                    pin_memory=True, 
                    collate_fn=contrastive_collate_fn
)
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
temperature = th.nn.parameter.Parameter(th.tensor([[0.50]], device=device), requires_grad=True)

opt = th.optim.Adam([
    {'params':model.parameters(), 'lr':1e-3},
    {'params':[temperature], 'lr':3e-2},
])
load_model(opt, str(OPTIMIZER_PATH))




# %%
step = 0
t_last_valid = 0#time.time()
t_last_save_all = time.time()
t_last_save_revert = time.time()

for ep in range(4*20):
    for ibatch, imgs in enumerate(train_dataloader):
        imgs = imgs.to(device)
        embeddings = model.embed(imgs)
        similarity_mat = F.normalize(embeddings) @ F.normalize(embeddings.transpose(0, 1))

        # since the ith row and ith column will always be 1
        # it brings implicit scale, to reduce it we put
        # 0 there (for softmax)
        zeroing_mat = get_zeroing_mat(len(imgs)//8, block_size=8).to(device)
        similarity_mat = similarity_mat.unsqueeze(0)*zeroing_mat
        similarity_mat2 = th.softmax(similarity_mat/temperature, dim=1)

        label_mat = contrastive_labels_mat(len(imgs)//8, block_size=8).to(device)

        loss = -(label_mat * th.log(similarity_mat2* + 1e-9)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()



        if ibatch%10 == 0:
            logger.report_scalar("loss", "train", iteration=step , value=loss.item())
            logger.report_scalar("temperature", "train", iteration=step , value=temperature.item())
            print(ep/4, step, loss.item(), temperature.item())

        if ibatch%100 == 0:
            logger.report_confusion_matrix("similarity mat", "0", iteration=step, matrix=similarity_mat2[0].detach().cpu().numpy())
            logger.report_confusion_matrix("similarity mat", "1", iteration=step, matrix=similarity_mat2[1].detach().cpu().numpy())
            logger.report_confusion_matrix("similarity mat", "2", iteration=step, matrix=similarity_mat2[2].detach().cpu().numpy())
        if time.time() - t_last_valid > VALID_TIME_DELTA:
            with th.no_grad():
                base_embeddings, base_labels = create_base_embeddings_and_labels(model, train_database_dataloader, device)
                average_great_circle_distance = eval_ranking_loop(model, valid_dataset, device, base_embeddings, base_labels, VALID_TEMPERATURE)
                logger.report_scalar("average_great_circle_distance", "valid", iteration=step , value=average_great_circle_distance)
            t_last_valid = time.time()

        if time.time() - t_last_save_all > SAVE_DELTA_ALL:
            save_model(model, str(MODEL_PATH))
            save_model(opt, str(OPTIMIZER_PATH))
            t_last_save_all = time.time()

        if time.time() - t_last_save_revert > SAVE_DELTA_REVERT:
            save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
            save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
            t_last_save_revert = time.time()
        

        step += 1


save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
