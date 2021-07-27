#!/bin/bash



# Minimal setup of Apache Arrow (C++ & Python) on FB's CentOS8 devservers

# This assumes that you have virtualenv, so you probably want to run
#   source f4d-deps/python-env/bin/activate
#
# if you're using system python on CentOS you might need to do:
#   sudo dnf install -y python36-devel

set -xe # better be safe than sorry

function with-proxy() {
  (
    export $(fwdproxy-config --format=sh curl)
    export http_no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    "$@"
  )
}

rm -rf arrow
git clone -b apache-arrow-3.0.0 https://github.com/apache/arrow.git

# adopted from https://arrow.apache.org/docs/developers/python.html#building-on-linux-and-macos

with-proxy pip install -r arrow/python/requirements-wheel-build.txt

# we need to install in the local directory because otherwise for some reason python build can't find the libarrow_python.so
export ARROW_HOME=$(pwd)/arrow-dist
export LD_LIBRARY_PATH=$(pwd)/arrow-dist/lib:$LD_LIBRARY_PATH
mkdir arrow/cpp/build
pushd arrow/cpp/build
cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME -DCMAKE_INSTALL_LIBDIR=lib -GNinja -DARROW_PYTHON=ON -DARROW_PARQUET=ON ..
with-proxy ninja -j 32
ninja install
popd

pushd arrow/python
with-proxy pip install wheel
export PYARROW_WITH_PARQUET=1
with-proxy python setup.py build_ext --bundle-arrow-cpp bdist_wheel
pip install dist/pyarrow-3.0.0-*.whl
popd
