from .Device import Device
import ctypes
import platform
import numpy as np
__all__ = ['mps']

device_functions = ctypes.CDLL("exclusiveAI/components/Device/libPyMetalBridge.dylib"),

device_functions[0].swift_sigmoid_derivative.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int
]

class mps(Device):
    def __init__(self):
        super().__init__(
            name="mps",
        )
    
    def sigmoid_derivative(self, input_array):
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_mutable_ptr = (ctypes.c_float * len(input_array))()
        device_functions[0].swift_sigmoid_derivative(input_ptr, output_mutable_ptr, len(input_array))
        return np.array(output_mutable_ptr)