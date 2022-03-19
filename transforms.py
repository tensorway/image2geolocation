from torchvision import transforms
import torch as th

train_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomResizedCrop(224),
        # transforms.ColorJitter(brightness=.5, hue=.3),
        # transforms.RandomHorizontalFlip(), commented out because we want the model to be able to use left right information
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.RandomErasing(),
        # transforms.RandomErasing(),
    ])

val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



def de_train_transform(img):
    img = img.permute(1, 2, 0)
    std = th.tensor([[[0.229, 0.224, 0.225]]])
    mu  = th.tensor([[[0.485, 0.456, 0.406]]])
    img = img*std + mu
    return img.detach().cpu().numpy()