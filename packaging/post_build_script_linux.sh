#!/bin/bash
set -ex

pip3 install auditwheel pkginfo

for pkg in dist/torchdata*.whl; do
    echo "PkgInfo of $pkg:"
    pkginfo $pkg

    auditwheel show $pkg
    auditwheel repair $pkg --plat manylinux2014_x86_64
done
