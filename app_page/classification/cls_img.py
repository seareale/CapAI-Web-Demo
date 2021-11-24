from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet
from utils.general import get_markdown

def run_cls_img():
    
    st.title("Transition Classification")
    model_type = frame_selector_ui()

    uploaded_file = st.file_uploader("Upload a image", ["jpg", "jpeg", "png"])
    net, device = load_model(model_type)
    label = {0:'06.stomach', 1:'07.intestineSS', 2:'08.intestineSF', 3:'09.intestineL'}

    if uploaded_file is not None:
        st.success("Success")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        img_path = f"data/{uploaded_file.name}"
        cv2.imwrite(img_path, opencv_image)
        img_org = cv2.imread(img_path)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

        st.markdown('## Input Image')  # display input image
        st.image(img_org)

        img = image_preprocess(img_org)
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = net(img)
            pred = output.argmax(dim=1)
            pred = pred.cpu().numpy()
            pred = int(pred[0])
            softmax = nn.Softmax(dim=1)
            st.markdown(f"## Predicted class: {label[pred]}")
            

            fig, ax = plt.subplots()
            x = range(len(softmax(output).cpu().numpy()[0]))
            ax.bar(x, softmax(output).cpu().numpy()[0]*100)
            ax.set_title("Predicted probability")
            ax.set_ylabel("Probability (%)")
            ax.set_xlabel("Class")
            ax.set_xticks([0,1,2,3])
            ax.set_xticklabels(['06.stomach','07.intestineSS','08.intestineSF','09.intestineL'])
            st.pyplot(fig)
        
           
    elif uploaded_file is None:
        st.info("Check the Image format (e.g. jpg, jpeg, png)")

def frame_selector_ui():
    model_list = list(Path("models/weights").glob("*.pth"))
    model_list = sorted([str(model.name)[:-4] for model in model_list])

    st.sidebar.markdown("# Options")

    model_type = st.sidebar.selectbox("Select model", model_list, 0)

    return model_type


def load_model(model_name="efficientnet"):
    device = torch.device("cuda:0")

    path = "models/weights/" + model_name + ".pth"


    if model_name == "efficientnet":
        net = EfficientNet.from_pretrained("efficientnet-b5", num_classes=4)
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["net"])
        net.eval()

    return net, device


def image_preprocess(img):

    transform_test = transforms.Compose(
                        [
                        transforms.ToPILImage(),
                        transforms.Resize(384),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]
                        )

    img = transform_test(img)
    return img