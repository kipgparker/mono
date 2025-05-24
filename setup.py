from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mono",
    version="0.1.0",
    author="kip",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="mono_cpp",  # This will be the name of the module when imported in Python
            sources=["csrc/binding.cpp", "csrc/cache.cu"],
            include_dirs=["csrc"],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    install_requires=[
        "torch>=2.4.1",
        "torchvision>=0.19.1",
        "torchaudio>=2.4.1",
        # "flash-attn",
    ],
    extras_require={
        "dev": [
            "black>=23.12.1",  # Latest stable version
        ],
    },
    python_requires=">=3.8"
) 