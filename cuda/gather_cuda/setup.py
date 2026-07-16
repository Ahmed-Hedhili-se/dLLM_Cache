from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gather_cuda",

    ext_modules=[
        CUDAExtension(
            name="gather_cuda",
            sources=[
                "gather.cpp",
                "gather_fp32.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],

    cmdclass={
        "BuildExtension": BuildExtension
    },
)