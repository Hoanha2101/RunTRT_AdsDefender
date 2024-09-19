from backbone import *
import torch

def convert_pytorch2onnx(model, input_samples, path_onnx, mode='float32bit', device='cuda'):
    if mode == 'float16bit':
        print("Converting model and inputs to float16")
        model = model.half()  # Convert model to float16
        input_samples = input_samples.half()  # Convert input samples to float16
    elif mode == 'float32bit':
        print("Converting model and inputs to float32")
        model = model.float()  # Convert model to float32
        input_samples = input_samples.float()  # Convert input samples to float32
    
    model.to(device)
    model.eval()
    input_samples = input_samples.to(device)
    
    torch.onnx.export(
        model,  # The model
        input_samples,  # Input tensor with desired size
        path_onnx,  # Path to save the ONNX file
        verbose=False,  # Whether to print the process
        opset_version=12,  # ONNX opset version
        do_constant_folding=True,  # Whether to do constant folding optimization
        input_names=['images'],  # Model input names
        output_names=['output'],  # Model output names
        dynamic_axes={
            'images': {0: 'batch_size'},  # Dynamic batch size for inputs
            'output': {0: 'batch_size'}  # Dynamic batch size for outputs
        }
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

#Load state dict
BACKBONE = MobileFaceNet(512)
state_dict = torch.load("models/org/model_mobilefacenet.pth", weights_only=True)
BACKBONE.load_state_dict(state_dict)

input_samples = torch.randn(1, 3, 112, 112)  # Example input tensor
path_onnx = "models/onnx/model_mobilefacenet_FP32.onnx"

# Convert the model to ONNX
convert_pytorch2onnx(BACKBONE, input_samples, path_onnx, mode='float32bit', device=device)

