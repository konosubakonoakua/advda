# from importlib.resources import path
# from pprint import pprint
from asyncio.log import logger

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
# from typing import List
import io
import numpy as np

# from numpy import short, source
# from requests import models
from backend.dto import Prediction
import torch
from fastapi import APIRouter
from backend import cfg
# import torchvision
# import base64
from backend.yolov5.models.common import Detections

router = APIRouter(prefix="/yolo", tags=["yolo"])

model_yolo = torch.hub.load(cfg.MODEL_REPO_YOLO, 'custom', path=cfg.MODEL_PATH_YOLO, source="local")
# model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg.MODEL_PATH_YOLO)
# model_yolo = model_yolo.half().eval().cuda()


# Define the /prediction route
@router.post('/prediction', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Resize image to expected input shape

        # Convert from RGBA to RGB *to avoid alpha channels*
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Convert image into grayscale *if expected*

        # Convert image into numpy format

        # Scale data (depending on your model)

        # Generate prediction
        res: Detections = model_yolo(pil_image)
        # res.display(pprint=True, show=True)
        return make_pred_response(file, res)

    except Exception as e:
        # e = sys.exc_info()[1]
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def make_pred_response(file: UploadFile, res: Detections) -> dict:
    clz = []
    bboxes = []
    confs = []
    if res.pred is not None:
        pred = res.pred[0]
        for *box, conf, cls in pred:  # xyxy, confidence, class
            bboxes.append({
                "x1": box[0].item(),
                "y1": box[1].item(),
                "x2": box[2].item(),
                "y2": box[3].item(),
            })
            clz.append({
                "id": int(cls.item()),
                "name": res.names[int(cls.item())]
            })
            confs.append(round(conf.item(), 2))

    imgs = res.render()  # updates results.imgs with boxes and labels
    img = Image.fromarray(imgs[0].astype(np.uint8))
    img_path = f"imgs/pred_{file.filename}"
    img.save(img_path)
    img_base64 = None
    # for img in res.imgs:
    #     buffered = io.BytesIO()
    #     img_base64 = Image.fromarray(img)
    #     img_base64.save(buffered, format="JPEG")
    #     img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')  # base64 encoded image with results

    return {
        'filename': file.filename,
        'contenttype': file.content_type,
        'boudingboxes': bboxes,
        'classes': clz,
        'confidences': confs,
        'image_with_bboxes': img_base64,
        'image_url': f"pred_{file.filename}"
    }


@router.get('/upload')
async def return_upload_page():
    with open("../web/static/yolo_upload.html", encoding='utf-8') as f:
        content = f.read()
    return HTMLResponse(content=content)
