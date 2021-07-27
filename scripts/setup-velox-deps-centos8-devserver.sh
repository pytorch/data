#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Minimal setup for FB's CentOS8 devservers

# You would need to run
#   source f4d-deps/python-env/bin/activate
# before any python work (e.g. before running `make format-check`

# You would need to run
#   source /opt/rh/gcc-toolset-9/enable
# before C++ build work

# Some of the packages must be build with the same compiler flags
# so that some low level types are the same size.
#
export COMPILER_FLAGS="-mavx2 -mfma -mavx -mf16c -masm=intel -mlzcnt"

set -e # better be safe than sorry

sudo rm -rf f4d-deps
mkdir  f4d-deps
cd     f4d-deps || exit 1

function with-proxy() {
  (
    export $(fwdproxy-config --format=sh curl)
    export http_no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    "$@"
  )
}

# Python3 refers to fbcode's python and it has pip disabled, let's activate it
python3.8 -m venv python-env
source python-env/bin/activate
with-proxy pip3 install cmake_format regex

# black and clang-format come preinstalled already

sudo dnf install -y cmake boost-devel double-conversion-devel glog-devel \
  snappy-devel fmt-devel libevent-devel libuuid-devel libunwind-devel \
  libdwarf-devel re2-devel

# hopefully can be removed in the future
sudo dnf install -y protobuf-devel lzo-devel

# ninja builds are faster!
sudo dnf install -y ninja-build

sudo dnf install -y gcc-toolset-9
source /opt/rh/gcc-toolset-9/enable

(
  git clone https://github.com/facebook/zstd &&
  cd zstd &&
  make -j 20 &&
  sudo make install -j 20
)
(
  git clone https://github.com/lz4/lz4 &&
  cd lz4 &&
  make -j 20 &&
  sudo make install -j 20
)

# On CentOS8 gtest is too old (1.8), we need at least 1.9
(
  wget $(fwdproxy-config wget) https://github.com/google/googletest/archive/release-1.10.0.tar.gz &&
  tar xf release-1.10.0.tar.gz &&
  cd googletest-release-1.10.0 &&
  cmake -GNinja -DCMAKE_CXX_STANDARD="$CXX_STANDARD" -DCMAKE_CXX_FLAGS="$COMPILER_FLAGS"  -DBUILD_SHARED_LIBS=ON . &&
  sudo ninja install
)

(
  wget $(fwdproxy-config wget) https://www.antlr.org/download/antlr4-cpp-runtime-4.8-source.zip
  mkdir antlr4-cpp-runtime-4.8-source
  cd antlr4-cpp-runtime-4.8-source
  unzip ../antlr4-cpp-runtime-4.8-source.zip
  mkdir build && mkdir run && cd build
  cmake ..
  DESTDIR=../run make install -j 20
  sudo cp -r ../run/usr/local/include/antlr4-runtime  /usr/local/include/.
  sudo cp ../run/usr/local/lib/*  /usr/local/lib/.
  sudo ldconfig
)

( git clone https://github.com/facebook/folly.git &&
  cd folly &&
  cmake -DCMAKE_CXX_FLAGS="$COMPILER_FLAGS" -DCMAKE_POSITION_INDEPENDENT_CODE=ON . &&
  sudo make install -j 20
)

#(
#  git clone https://github.com/ericniebler/range-v3.git &&
#  cd range-v3 &&
#  git checkout 0.11.0 &&
#  cmake . &&
#  make -j $NPROC &&
#  make install
#)
