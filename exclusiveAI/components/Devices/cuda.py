# import cupy 
from .Device import Device


class cuda(Device):
    def __init__(self):
        self.name = "cuda"
        self.device_id = 0
        # self.device = cupy.cuda.Device(self.device_id)
        self.device.use()
        self.device_properties = self.device.get_attributes()
        self.device_properties["name"] = self.device.name
        self.device_properties["compute_capability"] = self.device.compute_capability
        self.device_properties["total_memory"] = self.device.total_memory
        self.device_properties["memory_bandwidth"] = self.device.memory_bandwidth
        self.device_properties["clock_rate"] = self.device.clock_rate
        self.device_properties["multi_processor_count"] = self.device.multi_processor_count
        self.device_properties["max_threads_per_block"] = self.device.max_threads_per_block
        self.device_properties["max_block_size"] = self.device.max_block_size
        
    # def activationfn_sigmoid(self, v):
        # return cupy.sigmoid(v)