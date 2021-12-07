import cv2
import numpy as np
import torch
import torchvision

from models.general import non_max_suppression, scale_coords 
from models.yolo_format import letterbox
# config
IMG_SIZE = 384

LABELS = {
    0: "01_ulcer",
    1: "02_mass",
    2: "03_stricture",
    3: "04_lymph",
    4: "05_bleeding",
}

COLOR = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (122, 122, 0),
    4: (0, 122, 122),
    # 5: (122, 0, 122),
    # 6: (102, 34, 84),
}


def draw_image_with_boxes(image, pred, pred_img_shape, conf=0.6, iou=0.8):
    # Non-Maximum Suppression
    pred = non_max_suppression(
        pred,
        conf,
        iou,
        None,
        False,
        multi_label=False,
        max_det=140,
    )
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred = pred[0].cpu().detach().numpy()
    pred[:, :4] = scale_coords(pred_img_shape, pred[:, :4], image.shape)
    pred = np.concatenate([np.array([pred[:, 5]]).T, pred[:, :4]], axis=1)

    # get bboxes in original image
    result_img = image.copy()
    h, w = result_img.shape[:2]

    if len(pred) > 0:
        for bbox in pred:
            # draw bbox
            bbox = [int(x) for x in bbox]
            c1, c2 = (bbox[1], bbox[2]), (bbox[3], bbox[4])
            result_img = cv2.rectangle(result_img, c1, c2, COLOR[bbox[0]], 3)

            # text label
            fs = int((h + w) * np.ceil(1 ** 0.5) * 0.01)  # font size
            lw = round(fs / 10)
            tf = max(lw - 1, 1)  # font thickness
            text_w, text_h = cv2.getTextSize(LABELS[bbox[0]], 0, fontScale=lw / 3, thickness=tf)[0]
            c2 = c1[0] + text_w, c1[1] - text_h - 3
            result_img = cv2.rectangle(
                result_img, c1, c2, COLOR[bbox[0]], -1, cv2.LINE_AA
            )  # filled
            result_img = cv2.putText(
                result_img,
                LABELS[bbox[0]],
                (c1[0], c1[1] - 2),
                0,
                lw / 3,
                (255, 255, 255),
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return result_img


def preprocess_image(img, stride=32, half=True):
    # resize and padding
    img = letterbox(img, IMG_SIZE, stride=stride, auto=True)[0]

    # convert to tensor
    img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    # normalize and batch shape
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # fp16
    if half:
        img = img.half()
    return img
