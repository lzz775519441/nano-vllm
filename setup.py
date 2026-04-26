import os

from setuptools import setup


def get_extensions():
    if os.environ.get("NANOVLLM_BUILD_CUDA", "1") == "0":
        return [], {}
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception:
        return [], {}
    if CUDA_HOME is None:
        return [], {}

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]
    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": nvcc_flags,
    }
    ext_modules = [
        CUDAExtension(
            name="nanovllm._C",
            sources=[
                "nanovllm/csrc/bindings.cpp",
                "nanovllm/csrc/kernels.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
