# from importlib.resources import path
# from pprint import pprint
from asyncio.log import logger
import PIL

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
# from typing import List
import io
import numpy as np

# from numpy import short, source
# from requests import models
from backend.dto import PredictionClassifier
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet, resnet18, resnet50
from torch import nn
from fastapi import APIRouter
from backend import cfg
# import torchvision
# import base64
from backend.yolov5.models.common import Detections  # 仅作代码提示
from backend.utils.imagenet import get_imagenet_classes, get_imagenet_classes_zh


class ResnetPredictor(nn.Module):

    def __init__(self, model: ResNet):
        super().__init__()
        self.model = model
        self.transforms = nn.Sequential(
            # We use single int value inside a list due to torchscript type restrictions
            T.Resize([256, ]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            _ = self.model(x)
            y_pred: torch.Tensor = nn.Softmax(dim=1)(_).topk(10)
            confs = y_pred.values.clone().detach().cpu().tolist()
            clazz = y_pred.indices.clone().detach().cpu().tolist()
            # topk = y_pred.topk(5)
            return clazz, confs


router = APIRouter(prefix="/resnet", tags=["resnet"])

# model_yolo = torch.hub.load(cfg.MODEL_REPO_YOLO, 'custom', path=cfg.MODEL_PATH_YOLO, source="local")
# model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg.MODEL_PATH_YOLO)
# model_yolo = model_yolo.half().eval().cuda()
r18 = resnet18(pretrained=True, progress=True).eval().cuda()
r50 = resnet50(pretrained=True, progress=True).eval().cuda()
predictor_r50 = ResnetPredictor(r50).cuda()
predictor_r18 = ResnetPredictor(r18).cuda()
model_version_mapping = {
    50: predictor_r50,
    18: predictor_r18
}
classes_zh = get_imagenet_classes_zh()

# Define the /prediction route


@router.post('/{model_version}/prediction', response_model=PredictionClassifier)
async def prediction_route(model_version: int, file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        # pil_image = Image.open("./imgs/dogs.jpg")
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        img = T.ToTensor()(pil_image).unsqueeze_(0).cuda()

        # Generate prediction
        clazz, confs = model_version_mapping[model_version](img)
        torch.cuda.empty_cache()
        clazz_zh = [classes_zh[c] for c in clazz[0]]
        confs = [round(c, 5) for c in confs[0]]
        return make_pred_response(file, (clazz_zh, confs, clazz[0]))

    except Exception as e:
        # e = sys.exc_info()[1]
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def make_pred_response(file: UploadFile, res) -> dict:
    clz, confs, iids = res
    return {
        'filename': file.filename,
        'contenttype': file.content_type,
        'classes': clz,
        'iids': iids,
        'confidences': confs,
    }


# @router.get('/upload')
# async def return_upload_page():
#     with open("../web/static/yolo_upload.html", encoding='utf-8') as f:
#         content = f.read()
#     return HTMLResponse(content=content)
