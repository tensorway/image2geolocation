#%%
import torch as th
from torch import nn
from dataset import ClassificationDataset
from transforms import train_transform, val_transform


class BenchmarkModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2'):
        super().__init__()
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        if model_name == 'mobilenet_v2':
            self.model.classifier[1] = nn.Linear(1280, 21)
        else:   
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 21)
        self.model_name = model_name

    def embed(self, img):
        if self.model_name == 'mobilenet_v2':
            return self.model.features(img).mean(dim=-1).mean(dim=-1)
        tmp = self.model.fc
        self.model.fc = nn.Identity()
        embedding = self.model(img)
        self.model.fc = tmp
        return embedding

    def forward(self, img, device):
        return self.model(img)

# %%
if __name__ == '__main__':
    dataset = ClassificationDataset()
    model = BenchmarkModel()
    dic = dataset[0]
    for img in dic['images']:
        img.show()
    #print(train_transform(img))
    #print(train_transform(img).shape)
    batch = th.cat([train_transform(img).unsqueeze(0) for img in dic['images']], dim=0)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    preds = model(batch, device)
    print(preds)
    
# %%
