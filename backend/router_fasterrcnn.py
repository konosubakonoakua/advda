# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

from PIL import Image
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import argparse
import json
import yaml
import pprint

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import RobustDPatch

from backend.log import logger
from backend import cfg
from backend.dto import PredictionDetector
from backend.yolov5.models.common import Detections

from utils.coco import COCO_INSTANCE_CATEGORY_NAMES

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi import APIRouter
router = APIRouter(prefix="/fasterrcnn", tags=["fasterrcnn"])

model = None

# Define the /prediction route


@router.post('/prediction', response_model=PredictionDetector)
async def prediction_route(file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Resize image to expected input shape

        # Convert from RGBA to RGB *to avoid alpha channels*
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Generate prediction
        res: Detections = model(pil_image)
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


def extract_predictions(predictions_):

    # for key, item in predictions[0].items():
    #     print(key, item)

    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                         for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])]
                         for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.5
    predictions_t = [predictions_score.index(
        x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls):
    text_size = 5
    text_th = 5
    rect_th = 6

    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, boxes[i][0], boxes[i][1],
                      color=(0, 255, 0), thickness=rect_th)

        # Write the prediction class
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), thickness=text_th)

    plt.axis("off")
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()


def get_loss(frcnn, x, y):
    frcnn._model.train()
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    image_tensor_list = list()

    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            img = transform(x[i]).to(frcnn._device)
        image_tensor_list.append(img)

    loss = frcnn._model(image_tensor_list, y)
    for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss


def append_loss_history(loss_history, output):
    for loss in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss_history[loss] += [output[loss]]
    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False,
                        default=None, help="Path of config yaml file")
    cmdline = parser.parse_args()

    if cmdline.config and os.path.exists(cmdline.config):
        with open(cmdline.config, "r") as cf:
            config = yaml.safe_load(cf.read())
    else:
        config = {
            "attack_losses": ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
            "cuda_visible_devices": "1",
            "patch_shape": [450, 450, 3],
            "patch_location": [600, 750],
            "crop_range": [0, 0],
            "brightness_range": [1.0, 1.0],
            "rotation_weights": [1, 0, 0, 0],
            "sample_size": 1,
            "learning_rate": 1.0,
            "max_iter": 5000,
            "batch_size": 1,
            "image_file": "banner-diverse-group-of-people-2.jpg",
            "resume": False,
            "path": "xp/",
        }

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=False, attack_losses=config["attack_losses"], device_type=device_type
    )

    image_1 = cv2.imread(config["image_file"])
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_1 = cv2.resize(image_1, dsize=(
        image_1.shape[1], image_1.shape[0]), interpolation=cv2.INTER_CUBIC)

    image = np.stack([image_1], axis=0).astype(np.float32)

    attack = RobustDPatch(
        frcnn,
        patch_shape=config["patch_shape"],
        patch_location=config["patch_location"],
        crop_range=config["crop_range"],
        brightness_range=config["brightness_range"],
        rotation_weights=config["rotation_weights"],
        sample_size=config["sample_size"],
        learning_rate=config["learning_rate"],
        max_iter=1,
        batch_size=config["batch_size"],
    )

    x = image.copy()

    y = frcnn.predict(x=x)
    for i, y_i in enumerate(y):
        y[i]["boxes"] = torch.from_numpy(y_i["boxes"]).type(
            torch.float).to(frcnn._device)
        y[i]["labels"] = torch.from_numpy(
            y_i["labels"]).type(torch.int64).to(frcnn._device)
        y[i]["scores"] = torch.from_numpy(y_i["scores"]).to(frcnn._device)

    if config["resume"]:
        patch = np.load(os.path.join(config["path"], "patch.npy"))
        attack._patch = patch

        with open(os.path.join(config["path"], "loss_history.json"), "r") as file:
            loss_history = json.load(file)
    else:
        loss_history = {"loss_classifier": [], "loss_box_reg": [],
                        "loss_objectness": [], "loss_rpn_box_reg": []}

    for i in range(config["max_iter"]):
        print("Iteration:", i)
        patch = attack.generate(x)
        x_patch = attack.apply_patch(x)

        loss = get_loss(frcnn, x_patch, y)
        print(loss)
        loss_history = append_loss_history(loss_history, loss)

        with open(os.path.join(config["path"], "loss_history.json"), "w") as file:
            file.write(json.dumps(loss_history))

        np.save(os.path.join(config["path"], "patch"), attack._patch)
