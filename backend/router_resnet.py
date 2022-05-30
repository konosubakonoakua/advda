# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Optional
from backend.utils import get_imagenet_classes, get_imagenet_classes_zh
from backend.dto import PredictionClassifier
from backend import cfg
# from backend.log import logger
from backend.applog import logger
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import ResNet, resnet18, resnet50
import torchvision.transforms as T
import torchvision
import torch
import kornia
from kornia.enhance import denormalize
import numpy as np
from PIL import Image
import io
from fastapi.responses import HTMLResponse
from fastapi import File, UploadFile, HTTPException
from fastapi import APIRouter
import logging 

# from backend.yolov5.models.common import Detections  # 仅作代码提示

# logger
# logger = logging.getLogger('uvicorn')

# fastapi router
router = APIRouter(prefix="/resnet", tags=["resnet"])


img_preprocess = T.Compose([
    # We use single int value inside a list due to torchscript type restrictions
    T.Resize([256, ]),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
r18 = resnet18(pretrained=True, progress=True).eval().cuda()
r50 = resnet50(pretrained=True, progress=True).eval().cuda()
model_version_mapping = {
    50: r50,
    18: r18
}
classes_zh = get_imagenet_classes_zh()

class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unorm = UnNormalize()


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
        clazz, confs = get_classifier_pred(
            model_version_mapping[model_version], img, img_preprocess)
        del img
        torch.cuda.empty_cache()
        clazz_zh = [classes_zh[c].split(',')[0] for c in clazz[0]]
        confs = [round(c, 5) for c in confs[0]]
        return make_pred_response(file, (clazz_zh, confs, clazz[0], "null"))

    except Exception as e:
        # e = sys.exc_info()[1]
        # logger.error(f"{LINE()}: \n\t\t{e}")
        logger.exception("🚧", stacklevel=1)
        raise HTTPException(status_code=500, detail="🚧"+str(e))


@router.post('/{model_version}/attack/{attack_method}', response_model=PredictionClassifier)
async def attack_model(model_version: int, attack_method: str, file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.save(f"imgs/pred_r{model_version}_origin_{file.filename}")
        # pil_image = Image.open("./imgs/dogs.jpg")
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        img = img_preprocess(T.ToTensor()(pil_image)).unsqueeze_(0).cuda()
        # create the operator
        canny = kornia.filters.Canny(low_threshold=0.4, high_threshold=0.5)

        from timeit import default_timer as timer

        tic = timer()
        # blur the image
        x_magnitude, x_canny = canny(img.clone().detach().cpu())
        # logger.info(f"🚧 canny shape: {x_canny.shape} 🚧")
        img_canny: np.ndarray = kornia.tensor_to_image(x_canny)
        # logger.info(f"🚧 canny shape: {img_canny.shape} 🚧")
        # Generate prediction
        classifier = get_resnet_like_classifier(
            model_version_mapping[model_version])
        attack = FastGradientMethod(estimator=classifier, eps=0.8)
        
        # img = img.copy().cpu().numpy()
        x: np.ndarray = img.clone().detach().cpu().numpy()
        # mask = np.eye(224, 224)
        mask = np.zeros((3, 224, 224))
        # mask[:, 112:224] = 255
        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # adv = attack.generate(x=x, mask=img_canny)  # [b,c,h,w]

        toc = timer()
        logger.info(f"🚧 erfgsm time cost: {toc - tic} 🚧")
        adv = attack.generate(x=x, mask=mask)  # [b,c,h,w]
        clazz, confs = get_classifier_pred(
            model_version_mapping[model_version],
            torch.Tensor(adv).cuda()
        )
        # img = (255 * get_denormalized_imagenet(img.cpu())).squeeze_(0).numpy().transpose(1, 2, 0)
        img = unorm(img.cpu().squeeze_(0))
        # logger.info(f"from{__file__} : {img.shape}")
        img = T.ToPILImage()(img)
        # img = Image.fromarray(img.astype('uint8'), 'RGB')
        img_adv_path = f"imgs/pred_r{model_version}_resize_{file.filename}"
        img.save(img_adv_path)

        # img = (255 * get_denormalized_imagenet(adv.cpu())).squeeze_(0).numpy().transpose(1, 2, 0)
        img = unorm(torch.tensor(adv).squeeze_(0))
        # logger.info(f"from{__file__} : {img.shape}")
        # img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = T.ToPILImage()(img)
        img_adv_path = f"imgs/pred_r{model_version}_adv_{file.filename}"
        img.save(img_adv_path)

        img = Image.fromarray(kornia.tensor_to_image(x_canny.byte()*255))
        # logger.info(str(kornia.tensor_to_image(x_canny.byte()*255)))
        img.save(f"imgs/pred_r{model_version}_canny_{file.filename}")

        clazz_zh = [classes_zh[c].split(',')[0] for c in clazz[0]]
        confs = [round(c, 5) for c in confs[0]]
        return make_pred_response(file, (clazz_zh, confs, clazz[0], img_adv_path))

    except Exception as e:
        # e = sys.exc_info()[1]
        logger.exception("🚧", stacklevel=1)
        raise HTTPException(status_code=500, detail="🚧"+str(e))


def make_pred_response(file: UploadFile, res) -> dict:
    clz, confs, iids, urls = res
    return {
        'filename': file.filename,
        'contenttype': file.content_type,
        'classes': clz,
        'iids': iids,
        'confidences': confs,
        'advs': urls
    }


def get_resnet_like_classifier(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )
    return classifier


def get_classifier_pred(model: ResNet, x: torch.Tensor, t: Optional[T.Compose] = None) -> tuple:
    with torch.no_grad():
        if t is not None:
            x = t(x)
        _ = model(x)
        y_pred: torch.Tensor = nn.Softmax(dim=1)(_).topk(10)
        confs = y_pred.values.clone().detach().cpu().tolist()
        clazz = y_pred.indices.clone().detach().cpu().tolist()
        # topk = y_pred.topk(5)
        return clazz, confs


def get_denormalized_imagenet(img):
    # return img
    return denormalize(
        torch.Tensor(img),
        torch.Tensor([0.485, 0.456, 0.406]).unsqueeze_(0),
        torch.tensor([0.229, 0.224, 0.225]).unsqueeze_(0)
    )

