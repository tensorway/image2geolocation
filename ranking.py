#%%
import torch as th
from tqdm import tqdm
from transforms import val_transform
from utils import great_circle_distance



def contrastive_labels_mat(batch_size, block_size=4):
    '''
    generates a matrix in blocks of ones
    and with 0s on the diagonal
    example for a batch size of 3
    b 0 0
    0 b 0
    0 0 b 

    where b (block) and block_size=4:
    0 1 1 1
    1 0 1 1
    1 1 0 1
    1 1 1 0 
    '''
    mat = []
    for i in range(batch_size):
        for di in range(block_size):
            row = []
            for j in range(batch_size*block_size):
                if block_size*i <= j < block_size*(i+1) and block_size*i+di!=j:
                    row.append(1)
                else:
                    row.append(0)
            mat.append(row)
    return th.tensor(mat, dtype=th.float32).unsqueeze(0)


def get_zeroing_mat(batch_size, block_size):
    '''
    generates a matrix that zeroes out the 
    elements in a similarity matrix to create 'several'
    losses
    '''
    n = batch_size*block_size
    zeroing_mat = th.ones(block_size-1, n,n)
    for ib in range(block_size-1):
        for i in range(n):
            jblock_start = i//block_size * block_size
            idx = (i%block_size+ib+1)%block_size + jblock_start
            for j in range(block_size):
                if j+jblock_start != idx:
                    zeroing_mat[ib, i, j+jblock_start] = 0

    return zeroing_mat

def contrastive_labels_mat(batch_size, block_size):
    '''
    generates a matrix that zeroes out the 
    elements in a similarity matrix to create 'several'
    losses
    '''
    n = batch_size*block_size
    zeroing_mat = th.zeros(block_size-1, n,n)
    for ib in range(block_size-1):
        for i in range(n):
            jblock_start = i//block_size * block_size
            idx = (i%block_size+ib+1)%block_size + jblock_start
            for j in range(block_size):
                if j+jblock_start == idx:
                    zeroing_mat[ib, i, j+jblock_start] = 1

    return zeroing_mat



def create_base_embeddings_and_labels(model, dataloader, device):
    base_labels = []
    base_embeddings = []
    with th.no_grad():
        model.eval()
        for train_dict in tqdm(dataloader):
            imgs, labels = train_dict['images'], train_dict['labels']
            input = imgs[0].to(device)
            embeddings = model.embed(input)
            base_labels.append(labels)
            base_embeddings.append(embeddings)
        model.train()
    base_labels = th.cat(base_labels, dim=0).to(device)
    base_embeddings = th.cat(base_embeddings, dim=0)
    return base_embeddings, base_labels



def eval_ranking_loop(model, dataset, device, base_embeddings, base_labels, temperature=10):
    loop = tqdm(range(len(dataset)))
    model.eval()

    totdist = 0 
    for idx in loop:
        sample = dataset[idx]
        imgs, labels = sample['images'], sample['labels']
        with th.no_grad():
            batch = tuple(img.unsqueeze(0) for img in imgs)
            input = batch[0].to(device)
            preds = model.embed(input)
            scores = (base_embeddings@preds.transpose(0, 1)).to(device)
            preds = base_labels * th.softmax(scores / temperature, dim=0)
            preds = preds.sum(dim=0)
            # topsi = th.topk(scores[:, 0], k=100).indices.to(device)
            # preds = base_labels[topsi].mean(dim=0)

            totdist += great_circle_distance(preds, labels)
        loop.set_description(f"{totdist/(idx+1): 4.6f}")

    model.train()
    return totdist/(idx+1)


def transform270(image):
    image[:, :10, :10]   = th.ones(3, 10, 10)*th.tensor([[[1]], [[-0.5]], [[-0.5]] ] )
def transform180(image): 
    image[:, -10:, :10]  = th.ones(3, 10, 10)*th.tensor([[[-0.5]], [[1]], [[-0.5]] ] )
def transform90(image): 
    image[:, :10, -10:]  = th.ones(3, 10, 10)*th.tensor([[[-0.5]], [[-0.5]], [[1]] ] )
def transform0(image):
    image[:, -10:, -10:] = th.ones(3, 10, 10)*th.tensor([[[-1]], [[1]], [[0]] ] )