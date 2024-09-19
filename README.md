## Create Env

### python
```bash
conda create --name cuda python=3.9.18
```
### [pytorch cuda](https://pytorch.org/)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Activate env
```bash
conda activate cuda
```
### TensorRT
```bash
pip install tensorrt==8.6.1
pip install nvidia-pyindex
pip install nvidia-tensorrt
pip install cuda-python
pip install pycuda
```
## Convert
Convert `yolov8s-seg.pt` to `yolov8s-seg.engine` with option FP16
```bash
python convert_yolo_trt.py
```
Download model `mobilefacenet.pth` in [here](https://onedrive.live.com/?authkey=%21AIweh1IfiuF9vm4&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21836&parId=root&o=OneUp)

Download video in [here](https://www.youtube.com/watch?v=BorbTZoGCgI)

Now, I set up folder map
```bash
|data/
|-----|betway.png
|models/
|-----|org/
|---------|model_mobilefacenet.pth
|-----|onnx/
|---------|model_mobilefacenet_FP32.onnx
|-----|trt/
|---------|model_mobilefacenet_FP16.trt
|videos/
|-----|1.mp4
|__init__.py
|backbone.py
|convert_Mfacenet_onnx.py
|convert_Mfacenet_trt.py
|convert_yolo_trt.py
|infer.ipynb
|infer.py
|yolov8s-seg.engine
|yolov8s-seg.onnx
|yolov8s-seg.pt
```

Convert pytorch model to onnx for `model_mobilefacenet.pth`

```bash
python convert_Mfacenet_onnx.py
```
Convert onnx to TensorRT for `model_mobilefacenet_FP32.onnx`

```bash
python convert_Mfacenet_trt.py
```
## Infer

Run cell in `infer.ipynb `

or

```bash
python infer.py
```
