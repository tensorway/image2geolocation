#%%
import timm
import torch as th
from torch import nn
from dataset import Image2GeoDataset
from transforms import train_transform, val_transform


class BenchmarkModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2'):
        super().__init__()
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        if model_name == 'mobilenet_v2':
            self.model = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            self.model.classifier[1] = nn.Identity()
            nfeatures = 1280
        elif 'nvidia_efficientnet' in model_name:
            self.model = th.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=True)
            self.model.classifier[-1] = nn.Identity()
            nfeatures = 1792
        elif 'efficientnetv2' in model_name:
            self.model = timm.create_model(model_name, pretrained=True)
            nfeatures = self.model.classifier.weight.shape[1]
            self.model.classifier = nn.Identity()
        else:   
            self.model = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
            nfeatures = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        self.classifier = nn.Linear(nfeatures, 2)
        self.cov  = nn.Linear(nfeatures, 3)
        self.model_name = model_name

    def embed(self, img):
        return self.model(img)

    def load_state_dict(self, state_dict, okmore=True, okless=True):
        own_state = self.state_dict()
        cnt = 0
        for name, param in state_dict.items():
            if name not in own_state:
                message = 'therer are more parameters is passed state_dict than in self ' + name + ' does not exit'
                if okmore:
                    print(message)
                    continue
                raise RuntimeError(message) 
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            cnt += 1
        if cnt != len(own_state):
            message = 'not all parameters loaded, there are less parameters in passed state_dict than in self'
            if okless:
                print(message)
            else:
                raise RuntimeError(message) 


    def forward(self, img, device):
        embedded = self.embed(img)
        mean = self.classifier(embedded)+th.tensor([[44.475, 16.475]], device=device)  #adding to coverge faster
        cov = self.cov(embedded)

        #reshape, make symetric and with nonnegative diagonal elements
        cov = th.cat(( th.exp(cov[:, :1]), cov[:, 1:2], cov[:, 1:2] , th.exp(cov[:, 2:])), dim=1) 
        cov = cov.view(-1, 2, 2) + th.tensor([[[2, 0], [0, 2]]], device=device)
        return mean, cov

# %%
if __name__ == '__main__':
    dataset = Image2GeoDataset()
    model = BenchmarkModel('efficientnetv2_rw_m')
    dic = dataset[0]
    for img in dic['images']:
        img.show()
    print(train_transform(img))
    print(train_transform(img).shape)
    batch = th.cat([train_transform(img).unsqueeze(0) for img in dic['images']], dim=0)
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    model.to(device)
    batch = batch.to(device)
    print(model(batch, device))
# %%
