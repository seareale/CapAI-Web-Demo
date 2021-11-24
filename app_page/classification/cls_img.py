import streamlit as st
from utils.general import get_markdown
from pathlib import Path
from efficientnet_pytorch import EfficientNet
import torch
import torch.backends.cudnn as cudnn
import  cv2
import numpy as np 

def run_cls_img():
    readme_text = st.markdown(get_markdown("empty.md"), unsafe_allow_html=True)
    model_type = frame_selector_ui()
    st.markdown(model_type, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a image", ["jpg","jpeg","png"])
    net, device = load_model(model_type)
    st.write(uploaded_file)
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        img_path = f"data/classification/{uploaded_file.name}"
        cv2.imwrite(img_path, opencv_image)
        img_org = cv2.imread(img_path)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

        st.image(img_org)
    


def frame_selector_ui():
    model_list = list(Path("models/classification").glob("*.pth"))
    model_list = sorted([str(model.name)[:-4] for model in model_list])

    st.sidebar.markdown("# Options")

    model_type = st.sidebar.selectbox("Select model", model_list, 0)

    return model_type



def load_model(model_name="efficientnet"):
    device = torch.device("cuda:0")


    path = "models/classification/" + model_name + ".pth"

    if model_name == 'efficientnet':
        net = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'])
        net.eval()



    return net, device
