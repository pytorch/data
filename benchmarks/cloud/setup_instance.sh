# AMI: ami-0f18895ab5a90993c
# p3 instance

# Starting from AMI with driver
# Cuda is 11.0 by default

# Update CUDA Driver to 11.6
wget https://us.download.nvidia.com/tesla/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
sudo sh NVIDIA-Linux-x86_64-515.65.01.run

# Make sure PATH includes /usr/local/cuda-11.6/bin
#           LD_LIBRARY_PATH includes /usr/local/cuda-11.6/lib64
#           or, add /usr/local/cuda-11.6/lib64 to /etc/ld.so.conf and run ldconfig as root
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

# Install Nvidia cuDNN (Skip for now)
#wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz \
#     -O cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
#
#tar xJf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
#pushd cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive
#sudo cp include/* /usr/local/cuda/include
#sudo cp lib/* /usr/local/cuda/lib64
#popd
#sudo ldconfig


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
conda install -y -c pytorch magma-cuda116
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia
#conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# Install TorchData
sudo apt-get install git
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
wget https://torchdatabenchmarkdatasets.s3.amazonaws.com/CIFAR-10-images-master-repack.zip
unzip CIFAR-10-images-master-repack.zip


# Run Benchmark
python ~/data/benchmarks/torchvision_classification/train.py \
  --model mobilenet_v3_large --epochs 5 --batch-size 128 --workers 12 \
  --ds-type dp --fs custom --data-loader V2 --dataset cifar --output-dir ~/benchmark_outputs
