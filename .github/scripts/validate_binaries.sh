
AWS_ENABLED=1
if [[ ${MATRIX_PACKAGE_TYPE} = "conda" ]]; then
    conda install -y torchdata -c ${PYTORCH_CONDA_CHANNEL}
    if [[ ${TARGET_OS} = "windows" ]]; then
        AWS_ENABLED=0
    fi
else
    pip install ${PYTORCH_PIP_PREFIX} torchdata --extra-index-url ${PYTORCH_PIP_DOWNLOAD_URL}
fi

case "${AWS_ENABLED}" in
    "0")
        python ./test/smoke_test/smoke_test.py --no-s3
	;;
    "1")
        python ./test/smoke_test/smoke_test.py
	;;
    *)
        exit 1
	;;
esac
