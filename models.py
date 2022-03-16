#%%
import torch as th
from torch import nn
from dataset import Image2GeoDataset
from transforms import train_transform, val_transform


class BenchmarkModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2'):
        super().__init__()
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        if model_name == 'mobilenet_v2':
            self.model.classifier[1] = nn.Linear(1280, 2)
        elif 'resnet' in self.model_name:   
            num_ftrs = self.model.fc.in_features
            self.regressor = nn.Linear(num_ftrs, 2)
            self.model.fc = nn.Identity() 

        self.model_name = model_name

    def embed(self, img):
        if self.model_name == 'mobilenet_v2':
            return self.model.features(img).mean(dim=-1).mean(dim=-1)
        if 'resnet' in self.model_name:
            return self.model(img)

    def forward(self, img, device):
        if self.model_name == 'mobilenet_v2':
            return self.model(img) + th.tensor([[44.475, 16.475]], device=device) #adding to converge faster
        if 'resnet' in self.model_name:
            return self.regressor(self.model(img)) + th.tensor([[44.475, 16.475]], device=device) #adding to converge faster

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