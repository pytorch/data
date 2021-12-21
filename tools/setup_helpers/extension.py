from distutils.version import LooseVersion
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
from setuptools import Extension
from setuptools.command.build_ext import build_ext


__all__ = [
    'get_ext_modules',
    # 'CMakeExtension',
    'CMakeBuild',
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()

def get_ext_modules():
    return [
        Extension(name='torchdata._torchdata', sources=[])
    ]


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_PREFIX_PATH=' + os.environ['CMAKE_PREFIX_PATH'],
                      '-DCMAKE_CXX_FLAGS=' + "-fPIC"]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
