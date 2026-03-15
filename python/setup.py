"""
Setup script for pylynx — Python bindings for the LYNX DFT simulator.

Build:
    pip install .              # from python/ directory
    pip install -e .           # editable/development install

Or via CMake:
    cmake -DBUILD_PYTHON=ON ..
    make
"""

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.sep):
            extdir += os.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/lynx',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON=ON',
            '-DBUILD_TESTS=OFF',
        ]

        build_args = ['--config', 'Release']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMake configure from root of LYNX project
        lynx_root = os.path.dirname(ext.sourcedir)
        subprocess.check_call(
            ['cmake', lynx_root] + cmake_args,
            cwd=self.build_temp
        )

        # Build just the _core target
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', '_core'] + build_args,
            cwd=self.build_temp
        )


setup(
    name='pylynx',
    version='0.1.0',
    author='LYNX Developers',
    description='Python bindings for the LYNX real-space DFT simulator',
    long_description='Python/pybind11 interface to LYNX C++/CUDA DFT kernels.',
    packages=['lynx'],
    package_dir={'lynx': 'lynx'},
    ext_modules=[CMakeExtension('lynx._core', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.8',
    install_requires=['numpy'],
    extras_require={
        'ase': ['ase>=3.22'],
        'mpi': ['mpi4py>=3.0'],
    },
)
