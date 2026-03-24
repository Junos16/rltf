import psutil
import torch

def log_system_usage(prefix="[SYSTEM]"):
    ram = psutil.virtual_memory()
    print(f"{prefix} RAM Usage: {ram.percent}% ({ram.used / (1024**3):.2f}GB / {ram.total / (1024**3):.2f}GB)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"{prefix} GPU {i} ({torch.cuda.get_device_name(i)}) - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
