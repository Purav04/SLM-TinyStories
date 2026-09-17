from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="slm_cuda_kernels",
    ext_modules=[
        CUDAExtension(
            name="slm_cuda_kernels",
            sources=[
                "csrc/bindings.cpp",
                "csrc/matmul_kernel.cu",
                "csrc/softmax_kernel.cu",
                "csrc/layernorm_kernel.cu",
            ],
            extra_compile_args={
                # This torch build requires C++20 (its headers use concepts/
                # `requires`, `std::strong_ordering`, string_view::starts_with,
                # etc.), so pin it explicitly rather than relying on torch's
                # build backend to inject its own default. This needs GCC>=10
                # as the actual host compiler — on HPC images where the
                # system-default GCC is older (e.g. Rocky Linux 8's GCC 8.5),
                # `module load` a newer GCC (matching the CUDA module's build
                # compiler, e.g. gcc-13.2.0) before running pip install, or
                # you'll see "compare: No such file or directory" or
                # "unrecognized command line option -std=c++20" instead.
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
