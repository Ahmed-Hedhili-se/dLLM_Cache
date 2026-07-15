# In test.py
import os
import torch
from torch.utils.cpp_extension import load

# Get the directory where test.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Combine it to find the absolute path of gather_kernel.cu (or gather.cu)
cuda_src = os.path.join(current_dir, "gather_kernel.cu")  # <-- Verify this matches your .cu filename!

cuda_module = load(
    name="custom_gather", 
    sources=[cuda_src], 
    verbose=True
)