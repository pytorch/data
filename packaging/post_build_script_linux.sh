#!/bin/bash
set -ex

pip3 install auditwheel pkginfo

for pkg in dist/torchdata*.whl; do
    echo "PkgInfo of $pkg:"
    pkginfo $pkg

    auditwheel show $pkg
done
