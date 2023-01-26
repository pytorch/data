
if [[ ${MATRIX_PACKAGE_TYPE} = "conda" ]]; then
    conda install -y torchdata -c ${PYTORCH_CONDA_CHANNEL}
else
    pip install ${PYTORCH_PIP_PREFIX} torchdata --extra-index-url ${PYTORCH_PIP_DOWNLOAD_URL}
fi

python  ./test/smoke_test/smoke_test.py
