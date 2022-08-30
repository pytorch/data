# Use "ami-0f18895ab5a90993c" for CUDA driver

# Install Nvidia cuDNN
wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz \
     -O cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz

tar xJf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
pushd cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive
sudo cp include/* /usr/local/cuda/include
sudo cp lib/* /usr/local/cuda/lib64
popd
sudo ldconfig

# Install Dependencies
sudo apt-get install vim

# Install miniconda
CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base
# Use python3.8 by default
conda install -y python=3.8
conda install -y git-lfs cmake

# Install PyTorch
conda install -y -c pytorch magma-cuda113
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch-nightly

# Install TorchData
sudo yum update -y
sudo yum install git -y
git clone https://github.com/pytorch/data.git
cd data
git fetch origin pull/734/head:Benchmark
git checkout Benchmark
python setup.py develop

# Get data from S3
cd ~
mkdir benchmark_datasets
mkdir benchmark_outputs
cd benchmark_datasets
# Ideally use "aws s3 cp s3://torchdatabenchmarkdatasets/CIFAR-10-images-master-repack.zip CIFAR-10-images-master-repack.zip"
# TODO: Need to name "test" folder to "val"
wget https://torchdatabenchmarkdatasets.s3.amazonaws.com/CIFAR-10-images-master-repack.zip
unzip CIFAR-10-images-master-repack.zip
