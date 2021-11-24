import os
import sys
from pathlib import Path

import cv2
import numpy as np
import nvidia_smi
import streamlit as st
import torch

load_model_list = {}


def run_det_vid():
    # side bar
    model_type, conf_slider, iou_slider = frame_selector_ui()

    # model init
    model, device = load_model(model_name=model_type)

    # file upload
    uploaded_vid = st.file_uploader("Upload a video", ["mp4"])
    # cap = cv2.VideoCapture(uploaded_vid)

    # tfile = tempfile.NamedTemporaryFile(delete=False)
    # tfile.write(f.read())

    # vf = cv.VideoCapture(tfile.name)

    # stframe = st.empty()

    # while vf.isOpened():
    #     ret, frame = vf.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     stframe.image(gray)

    # dvide container into two parts
    _, col, _ = st.columns([1, 8, 1])

    if uploaded_vid is not None:  # inference
        bytes_data = uploaded_vid.getvalue()
        decoded = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)

        # load a image
        os.makedirs("data", exist_ok=True)
        img_path = f"data/{uploaded_vid.name}"
        cv2.imwrite(img_path, decoded)
        img_org = cv2.imread(img_path)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        with col:
            st.markdown('**<div align="center">Input image</div>**', unsafe_allow_html=True)
            st.image(img_org, use_column_width=True)  # display input image
            st.markdown('<br/>**<div align="center">Output image</div>**', unsafe_allow_html=True)

        ###############################################################################
        # inference results
        with torch.no_grad():
            if model_type == "yolov5":
                from models.yolov5 import yolov5

                # image preprocessing
                img = yolov5.preprocess_image(img_org, stride=int(model.stride.max()))

                # inferencem
                pred = model(img.to(device))[0]
                img_bboxes = yolov5.draw_image_with_boxes(
                    img_org, pred, img.shape[2:], conf=conf_slider, iou=iou_slider
                )  # get bboxes and labels
            elif model_type == "swin_htc":
                from mmdet.apis import inference_detector

                pred = inference_detector(model, img_path)
                img_bboxes = model.show_result(img_path, pred, score_thr=conf_slider)
                img_bboxes = cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB)
            else:
                pass
        ###############################################################################

        with col:
            st.image(img_bboxes, use_column_width=True)  # display input image

    elif uploaded_vid is None:
        st.info("Check the Image format (e.g. mp4)")


def frame_selector_ui():
    model_list = list(Path("models/weights").glob("*.pt"))
    model_list = sorted([str(model.name)[:-3] for model in model_list])

    st.sidebar.markdown("# Options")

    model_type = st.sidebar.selectbox("Select model", model_list, 1)

    conf_slider = st.sidebar.slider(
        "conf threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01
    )

    if model_type == "yolov5":
        iou_slider = st.sidebar.slider(
            "IoU threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01
        )
    else:
        iou_slider = None

    return model_type, conf_slider, iou_slider


def load_model(model_name="yolov5", half=True):
    device = torch.device("cuda:0")

    if model_name in load_model_list.keys():
        return load_model_list[model_name], device

    path = "models/weights/" + model_name + ".pt"

    ###

    load_model_list[model_name] = model

    return model, device
