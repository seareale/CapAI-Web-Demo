import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import nvidia_smi
import streamlit as st
import torch

load_model_list = {}
use_model_list = {}


def run_det_vid():
    # side bar
    model_type, conf_slider, iou_slider = frame_selector_ui()

    # model init
    model, device = load_model(model_name=model_type)

    # file upload
    uploaded_vid = st.file_uploader("Upload a video", ["mp4"])

    # dvide container into two parts
    _, col1, col2, _ = st.columns([1, 4, 4, 1])

    if uploaded_vid is not None:  # inference
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        vid_org_path = tfile.name

        with col1:
            st.markdown('**<div align="center">Input video</div>**', unsafe_allow_html=True)
            st.video(vid_org_path)  # display input image
        with col2:
            st.markdown('**<div align="center">Output video</div>**', unsafe_allow_html=True)
            videobox = st.empty()

        ###############################################################################
        # inference results
        vid_org = cv2.VideoCapture(vid_org_path)

        while use_model_list[model_type]:
            time.sleep(0.1)
            videobox.warning("Model is in use!\n Please wait...")
        videobox.empty()
        # videobox.image("loading.png")

        use_model_list[model_type] = True

        while True:
            end_flag, frame_org = vid_org.read()
            if not end_flag:
                break

            with torch.no_grad():
                if model_type == "yolov5":
                    from models.yolov5 import yolov5

                    # image preprocessing
                    frame = yolov5.preprocess_image(frame_org, stride=int(model.stride.max()))

                    # inferencem
                    pred = model(frame.to(device))[0]
                    frame_bboxes = yolov5.draw_image_with_boxes(
                        frame_org, pred, frame.shape[2:], conf=conf_slider, iou=iou_slider
                    )  # get bboxes and labels
                elif model_type == "swin_htc":
                    from mmdet.apis import inference_detector

                    pred = inference_detector(model, frame)
                    img_bboxes = model.show_result(frame, pred, score_thr=conf_slider)
                    img_bboxes = cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB)
                else:
                    pass
            with col2:
                videobox.image(frame_bboxes, channels="BGR")

        use_model_list[model_type] = False
        ###############################################################################

        # with col2:
        #     st.video(vid_org_path)  # display input image

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

    # for check GPU Memory
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # load model
    MEGABYTES = 2.0 ** 20.0
    weights_warning, progress_bar = None, None
    used_memory_temp = 0
    try:
        weights_warning = st.warning("Loading %s..." % path)
        progress_bar = st.progress(0)

        # for import model library
        sys.path.append(f"./models")
        sys.path.append(f"./models/{model_name}")

        ###############################################################################
        if model_name == "yolov5":
            model = torch.load(path, map_location=device)["model"].float()
        elif model_name == "swin_htc":
            from mmcv import Config
            from mmdet.apis import init_detector

            config = "models/swin_htc/htc_swin_cascade_fpn.py"
            classes = Config.fromfile(config).classes
            model = init_detector(config, path, device=device)
            model.CLASSES = classes
        else:
            pass
            # model = torch.load(path, map_location=device).float()
        ###############################################################################

        while True:
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            used_memory = info.used / MEGABYTES
            total_memory = info.free / MEGABYTES

            if used_memory_temp == used_memory:
                break

            used_memory_temp = used_memory

            # We perform animation by overwriting the elements.
            weights_warning.warning(
                "Loading %s... (%6.2f/%6.2f MB)" % (path, used_memory, total_memory)
            )
            progress_bar.progress(min(used_memory / total_memory, 1.0))

    finally:
        ###############################################################################
        if model_name == "yolov5":
            from models.yolov5 import yolov5

            # for inference
            model.eval()
            if half:
                model.half()
            # for warming up
            model(
                torch.zeros(1, 3, yolov5.IMG_SIZE, yolov5.IMG_SIZE)
                .to(device)
                .type_as(next(model.parameters()))
            )
        ###############################################################################

        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

    load_model_list[model_name] = model
    use_model_list[model_name] = False

    return model, device
