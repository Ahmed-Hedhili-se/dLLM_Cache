from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="scatter_cuda",
    ext_modules=[
        CUDAExtension(
            name="scatter_cuda",
            sources=["scatter.cpp","scatter_fp32.cu",],
            extra_compile_args={
                "cxx": ["-O3",],
                "nvcc": ["-O3","--use_fast_math",],
                
                
                },

        
        )
    
    
    ],
    cmdclass={
        "BuildExtension": BuildExtension
    },
)