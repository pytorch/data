#!/bin/bash
set -ex

pip3 install auditwheel pkginfo

for pkg in dist/torchdata*.whl; do
    echo "PkgInfo of $pkg:"
    pkginfo $pkg

    auditwheel repair $pkg --plat manylinux2014_x86_64 -w wheelhouse

    pkg_name=`basename ${pkg%-linux_x86_64.whl}`
    auditwheel show wheelhouse/${pkg_name}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
done
