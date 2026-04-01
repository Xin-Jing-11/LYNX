"""
Setup script for pylynx — Python bindings for the LYNX DFT simulator.

Build:
    pip install .              # from python/ directory
    pip install -e .           # editable/development install (requires pre-built .so)

Recommended development workflow:
    cd build && cmake .. -DBUILD_PYTHON=ON && make -j$(nproc)
    cd ../python && pip install -e .
"""

import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Determine where setuptools expects the .so to be placed
        ext_fullpath = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_fullpath)
        os.makedirs(ext_dir, exist_ok=True)

        # Check if the .so is already built in lynx/ (from CMake build)
        lynx_dir = os.path.join(ext.sourcedir, 'lynx')
        existing_so = glob.glob(os.path.join(lynx_dir, '_core.cpython*.so'))

        if existing_so:
            # Copy the pre-built .so to where setuptools expects it
            shutil.copy2(existing_so[0], ext_fullpath)
            return

        # No pre-built .so — build from source via CMake
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON=ON',
            '-DBUILD_TESTS=OFF',
        ]

        # Pass through LYNX build options from environment
        for env_var, cmake_var in [
            ('USE_CUDA', '-DUSE_CUDA=ON'),
            ('USE_MKL', '-DUSE_MKL=ON'),
        ]:
            if os.environ.get(env_var, '').lower() in ('1', 'on', 'true', 'yes'):
                cmake_args.append(cmake_var)

        build_args = ['--config', 'Release', '-j']

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
    ext_modules=[CMakeExtension('lynx._core', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
)
