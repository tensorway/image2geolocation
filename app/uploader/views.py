from numpy import average
from rest_framework.response import Response
from rest_framework.views import APIView

import sys

sys.path.append("../")

from PIL import Image
import torch
import torch as th
from pathlib import Path
import torch.nn.functional as F
from models import BenchmarkModel
from dataset import Image2GeoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transforms import train_transform, val_transform
from utils import (
    load_model,
    save_model,
    great_circle_distance,
    seed_everything,
    draw_prediction,
)

MODEL_CHECKPOINTS_PATH = Path("model_checkpoints/")
MODEL_NAME = "mobilenetv2_benchmark"
MODEL_NAME = "resnet50_benchmark"
MODEL_NAME = "resnet152_benchmark"
MODEL_NAME = "efficientnetb4"
# Create your views here.

MODEL_PATH = MODEL_CHECKPOINTS_PATH / ("model_" + MODEL_NAME + ".pt")
THE_SEED = 42
TRAIN_DATA_FRACTION = 0.85

seed_everything(THE_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

model = BenchmarkModel()
load_model(model, str(MODEL_PATH))
model.to(device)


class UploadImagesApi(APIView):

    permission_classes = ()

    def post(self, request, *args, **kwargs):

        files = request.FILES.getlist("files")
        if not files:
            return Response({"message": "images_not_found"}, 404)

        imgs = []

        for img_ in files:
            imgs.append(Image.open(img_))

        with th.no_grad():
            batch = tuple(val_transform(img).unsqueeze(0) for img in imgs)
            batch = th.cat(batch, dim=0).to(device)
            preds = model(batch, device)

        predictions = preds.detach().cpu().numpy()
        averaged_predictions = predictions.mean(axis=0)
        return Response(
            {"latitude": averaged_predictions[0], "longitude": averaged_predictions[1]},
            200,
        )

