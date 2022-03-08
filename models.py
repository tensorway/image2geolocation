#%%
import torch as th
from torch import nn
from dataset import ClassificationDataset
from transforms import train_transform, val_transform



class BenchmarkModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2', NUM_OF_CLASSES=4):
        super().__init__()
        th.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = th.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        if model_name == 'mobilenet_v2':
            self.model.classifier[1] = nn.Linear(1280, NUM_OF_CLASSES)
        else:   
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, NUM_OF_CLASSES)
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
        x = self.model(img)
        return th.softmax(x,dim=-1).unsqueeze(0)

# %%
if __name__ == '__main__':
    dataset = ClassificationDataset()
    model = BenchmarkModel(NUM_OF_CLASSES=dataset.number_of_classes())
    dic = dataset[0]
    for img in dic['images']:
        img.show()
    #print(train_transform(img))
    #print(train_transform(img).shape)
    batch = th.cat([train_transform(img).unsqueeze(0) for img in dic['images']], dim=0)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    preds = model(batch, device)
    print(preds.size())
    
# %%
