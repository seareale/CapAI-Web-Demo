# web-demo-streamlit

<br/>

This project demonstrates object detection models([YOLOv5](https://github.com/ultralytics/yolov5), [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)) and classification model([EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)) into an interactive [Streamlit](https://streamlit.io) app.

<br/>

## Run
```bash
# To run this demo
$ git clone https://github.com/seareale/torch_deploy.git
$ cd torch_deploy
$ pip install -r requirements.txt
$ streamlit run main.py
```

## For inference
```bash
# for swin_htc
$ pip install openmim
$ mim install mmdet

# for yolov5
$ pip install -r requirements.txt
```

<br/>

---

<br/>

**<div align="center">made by [Seareale](https://github.com/seareale), [Jaebbb](https://github.com/jaebbb) | [KVL Lab](http://vl.knu.ac.kr) | [KNU](http://knu.ac.kr)</div>**

