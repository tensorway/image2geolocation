#%%
import torch as th
from torch import nn
from dataset import Image2GeoDataset
import timm
from transforms import train_transform, val_transform
import torch.nn.functional as F


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
        self.embed1 = nn.Linear(nfeatures, nfeatures)
        self.embed2 = nn.Linear(nfeatures, nfeatures)
        self.embed3 = nn.Linear(nfeatures, nfeatures)
        self.model_name = model_name

    def forward(self, img, device):
        return self.classifier(self.model(img)) + th.tensor([[44.475, 16.475]], device=device) #adding to converge faster

    def embed(self, img):
        x = self.model(img)
        x = F.relu(self.embed1(x))
        x = F.relu(self.embed2(x))
        return F.normalize( self.embed2(x) )


# %%
if __name__ == '__main__':
    dataset = Image2GeoDataset()
    model = BenchmarkModel()
    dic = dataset[0]
    for img in dic['images']:
        img.show()
    print(train_transform(img))
    print(train_transform(img).shape)
    batch = th.cat([train_transform(img).unsqueeze(0) for img in dic['images']], dim=0)
    print(model(batch))
# %%