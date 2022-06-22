@REM Copyright (c) Meta Platforms, Inc. and affiliates.
@REM All rights reserved.
@REM
@REM This source code is licensed under the BSD-style license found in the
@REM LICENSE file in the root directory of this source tree.

@echo off

git config --system core.longpaths true

git submodule update --init --recursive
if errorlevel 1 exit /b 1

python setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit /b 1
