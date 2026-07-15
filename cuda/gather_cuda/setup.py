from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name="gather_cuda",
    ext_modules=[
        CUDAExtension(
            name="gather_cuda",
            sources=[
                "gather.cpp",
                "gather.cu"
            ],
        )
    ],
    cmdclass={
        "BuildExtension": BuildExtension
    }
)