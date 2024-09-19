from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-seg.pt")

# Export the model
model.export(format="engine", half = True) # half = True -> FP16 | False -> FP32