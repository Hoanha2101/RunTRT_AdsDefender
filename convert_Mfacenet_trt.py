import os
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def initialize_builder(use_fp16=False, workspace_size=(1 << 31)):  # 2GB expressed using bit shift
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    config.max_workspace_size = workspace_size  # 2GB using bit shift

    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    return builder, config

def parse_onnx_model_static(builder, onnx_file_path, batch_size=2):
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("✅ Completed parsing ONNX file")
    
    # Thiết lập kích thước batch cố định cho tất cả các input
    for i in range(network.num_inputs):
        shape = list(network.get_input(i).shape)
        shape[0] = batch_size
        network.get_input(i).shape = shape

    return network

def set_dynamic_shapes(builder, config, dynamic_shapes):
    if dynamic_shapes:
        print(f"===> Using dynamic shapes: {str(dynamic_shapes)}")
        profile = builder.create_optimization_profile()

        for binding_name, dynamic_shape in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(binding_name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)
        
def build_and_save_engine(builder, network, config, engine_file_path):
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception as e:
            print(f"Cannot remove existing file: {engine_file_path}. Error: {e}")

    print("Creating TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"===> Serialized Engine Saved at: {engine_file_path}")
    else:
        print("❌ Failed to build engine")

# Fix batch_size
def main(onnx_file_path,engine_file_path, use_fp16 = False):
    batch_size = 1

    builder, config = initialize_builder(use_fp16=True)
    network = parse_onnx_model_static(builder, onnx_file_path, batch_size=batch_size)
    if network:
        build_and_save_engine(builder, network, config, engine_file_path)
        
if __name__ == "__main__":
    onnx_file_path = "models/onnx/model_mobilefacenet_FP32.onnx"
    engine_file_path = "models/trt/model_mobilefacenet_FP16.trt"
    use_fp16 = True
    main(onnx_file_path,engine_file_path, use_fp16)