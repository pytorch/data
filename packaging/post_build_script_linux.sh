#!/bin/bash
set -ex

cpu_arch=`uname -m`

pip3 install auditwheel pkginfo

for pkg in dist/torchdata*.whl; do
    echo "PkgInfo of $pkg:"
    pkginfo $pkg

    auditwheel repair $pkg --plat manylinux2014_${cpu_arch} -w wheelhouse

    pkg_name=`basename ${pkg%-linux_${cpu_arch}.whl}`
    auditwheel show wheelhouse/${pkg_name}-manylinux_2_17_${cpu_arch}.manylinux2014_${cpu_arch}.whl
done
