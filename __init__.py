import tensorrt as trt
import pycuda.driver as cuda

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem) -> None:
        self.host = host_mem
        self.device = device_mem
    

    def __str__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        return self.__str__()
    
class TensorrtBase:
    def __init__(self, engine_file_path, input_names,  output_names, *, gpu_id=0, dynamic_factor=1, max_batch_size=1) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        self.max_batch_size = max_batch_size
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)

            
    def _load_engine(self, engine_file_path):
        # Force init TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
        
    
    def _allocate_buffer(self, dynamic_factor):
        """Allocate buffer
        :dynamic_factor: normally expand the buffer size for dynamic shape
        """
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()
        for binding in self.binding_names:
            binding_idx = self.engine[binding]
            if binding_idx == -1:
                print("❌ Binding Names!")
                continue

            # trt.volume() return negtive volue if -1 in shape
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * \
                    self.max_batch_size * dynamic_factor
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings[binding_idx] = int(device_mem)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    def __del__(self):
        self.cuda_ctx.pop()
        del self.cuda_ctx