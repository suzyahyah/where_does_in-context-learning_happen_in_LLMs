#!/usr/bin/python3
# Author: Suzanna Sia


import torch
import pynvml
import os
import numpy as np

def gpu_util(func):
    # this is a decorator
    def fn_wrapper(*arg, **kwargs):
        result = func(*arg, **kwargs)
        output = print_gpu_utilization()
        print(f"Function {func.__name__} --> {output}")
        return result
    return fn_wrapper
    
def print_gpu_utilization():

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch_gpu_id = torch.cuda.current_device()
        pynvml.nvmlInit()
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_gpu_id = int(devices[torch_gpu_id])
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info_used = info.used//1024**2
        info_total = info.total//1024**2

        text = f"GPU {nvml_gpu_id} memory occupied: {info_used}/{info_total} MB\
                = {np.round((info_used*100)/info_total,2)}%."

        pynvml.nvmlShutdown()
    else:
        text = "No visible cuda devices"
    print(text)
    return text



def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
