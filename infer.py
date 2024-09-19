import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from __init__ import TensorrtBase
from ultralytics import YOLO
import time
from numpy import dot
from numpy.linalg import norm


input_names = ['images']
output_names = ['output']
batch = 1

net = TensorrtBase("models/trt/model_mobilefacenet_FP16.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch)

model_seg = YOLO("yolov8s-seg.engine")

def INFER_TRT(images):
    images = np.expand_dims(images, axis=0)
    images = np.ascontiguousarray(images).astype(np.float32)
    net.cuda_ctx.push()
    inputs, outputs, bindings, stream = net.buffers
    # Set optimization profile and input shape
    net.context.set_optimization_profile_async(0, stream.handle)
    net.context.set_input_shape("x", images.shape)
    
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(inputs[0].device, images, stream)
    # Execute inference
    net.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)      
    # Transfer predictions back to the host
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    stream.synchronize()
    
    # Copy outputs
    trt_outputs = [out.host.copy() for out in outputs]
    net.cuda_ctx.pop()
    return trt_outputs[0]

THRESHOLD = 0.7

# Open video capture
video = cv2.VideoCapture("videos/1.mp4")
img1 = cv2.resize(cv2.imread("data/betway.png"), (112,112))

example_embedding = INFER_TRT(img1)
VEC_refer = [example_embedding for i in range(20)]
print("Number of bet logos for reference: ",len(VEC_refer))

while True:
    s = time.time()
    ret, frame = video.read()
    
    if not ret:
        break
    results = model_seg(frame, classes=0, device="cuda", verbose=False)
    
    for result in results:
        if result.masks is not None:
            List_sim = []
            for mask, box in zip(result.masks.xy, result.boxes.xyxy):
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_img = frame[ymin:ymax, xmin:xmax]
                cropped_img = cv2.resize(cropped_img, (112, 112))
                out_infer_trt = INFER_TRT(cropped_img)
                
                for i in VEC_refer:
                    cos_sim = dot(i, out_infer_trt)/(norm(i)*norm(out_infer_trt))
                    List_sim.append(cos_sim)
                    
                if max(List_sim) >= THRESHOLD:
                    points = np.int32([mask])
                    points = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(frame, points, (255,255,255))
                          
    elapsed_time = time.time() - s
    if elapsed_time > 0:
        fps = 1 / elapsed_time
        # print(f"FPS: {round(fps, 5)}")
    else:
        continue
    
    cv2.putText(frame, str(round(fps,5)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 5, cv2.LINE_AA)
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()