from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nanovllm",
    ext_modules=[
        CUDAExtension(
            name="nanovllm._C",
            sources=[
                "nanovllm/csrc/bindings.cpp",
                "nanovllm/csrc/kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
            extra_link_args=["-Wl,--no-as-needed", "-lcuda"]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)