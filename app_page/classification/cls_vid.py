from pathlib import Path

import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch import nn
from utils.general import get_markdown, plot_confusion_matrix
from utils.make_dataset import CapAI

from glob import glob 
from collections import Counter
from sklearn.metrics import confusion_matrix
from PIL import Image
import GPUtil
import time

load_model_list = {}

def run_cls_default():
    st.title("Transition Classification")
    model_type = frame_selector_ui()

    net, device = load_model(model_type)
    label_names = ["06.stomach", "07.intestineSS", "08.intestineSF", "09.intestineL"]

    Path = 'data/testset/'


    col1, col2 = st.columns([1, 2])
    with col1:
        position = st.radio('Select the position', ['ALL', '06.stomach', '07.intestineSS', "08.intestineSF", "09.intestineL"], key=1)
    
    if position:
        with col2:
            if position == "ALL":
                position = '*'
            x = glob(f"{Path}{position}/*")
            patientid = sorted(list(set([x[i].split('/')[-1] for i in range(len(x))])))
            patientid.insert(0, "ALL")
            multi_select = st.multiselect('Please select patientID in multi selectbox!',
                                patientid)
    

    if position and multi_select:
        if position == "ALL":
            position = '*'


        if "ALL" not in multi_select:
            x = [x[i].replace(Path,'') for i in range(len(x)) if x[i].split('/')[-1] in multi_select]
        else:
            x = [x[i].replace(Path,'') for i in range(len(x))]

        x = sorted(x)
        st.markdown('---')

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('## 선택된 이미지 LIST')
            st.write(x)
        with col2:
            st.markdown('## Visualize Images')
            option = st.selectbox('', x)
            if option:
                imgs = sorted(glob(f"{Path}{option}/*.jpg"))
                fig, ax = plt.subplots(2, 5, figsize=(10, 5))
                for i, ax in enumerate(fig.axes):
                    ax.imshow(cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2RGB))
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                st.pyplot(fig)

        
        st.markdown('---')
        st.markdown('## Output')

        testset = CapAI(x)
        testloader = torch.utils.data.DataLoader(
                testset, batch_size=10, shuffle=False, num_workers=2)

        with torch.no_grad():
            y_true = []
            y_pred = []

            for i, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                y_true.append(list(np.array(labels.cpu()))[0])
                y_pred.append(Counter(list(predicted.cpu().numpy())).most_common()[0][0])
            
            CM = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(CM, target_names=label_names)
            image = Image.open('Confusion_matrix.png')
            st.image(image)
            os.remove('Confusion_matrix.png')

        st.markdown('---')

       


        '''
        
        fig, ax = plt.subplots()
        GPUs = GPUtil.getGPUs()
        x = [0, 1, 2, 3]
        y = [int(gpu.load*100) for gpu in GPUs]
        bar = ax.bar(x, y)
        the_plot = st.pyplot(fig)
        time.sleep(1)
        while True:
            GPUs = GPUtil.getGPUs()
            x = [0, 1, 2, 3]
            y = [int(gpu.load*100) for gpu in GPUs]
            bar.remove()
            bar = ax.bar(x,y)
            the_plot.pyplot(fig)
            time.sleep(1)
       
        '''


    


def frame_selector_ui():
    model_list = list(Path("models/weights").glob("*.pth"))
    model_list = sorted([str(model.name)[:-4] for model in model_list])

    st.sidebar.markdown("# Options")

    model_type = st.sidebar.selectbox("Select model", model_list, 0)

    return model_type


def load_model(model_name="efficientnet"):
    device = torch.device("cuda:0")

    path = "models/weights/" + model_name + ".pth"

    if model_name in load_model_list:
        return load_model_list[model_name], device

    if model_name == "efficientnet":
        net = EfficientNet.from_pretrained("efficientnet-b5", num_classes=4)
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["net"])
        net.eval()
    
    load_model_list[model_name] = net

    return net, device

