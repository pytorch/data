This folder contains templates that are useful for cloud setups

Idea would be to provision a machine by configuring it in a YAML file and then running a benchmark script on it
automatically. This is critical both for ad hoc benchmarking that are reproducible but also including real world
benchmarks in a release.

We've provided some useful `yml` templates for you to get started

https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-cli-creating-stack.html

## Setup aws cli

`aws configure` and enter your credentials

## Setup stack (machine configuration)

```sh
 aws cloudformation create-stack \
  --stack-name torchdatabenchmark \
  --template-body ec2.yml \
  --parameters ParameterKey=InstanceTypeParameter,ParameterValue=p3.2xlarge ParameterKey=DiskType,ParameterValue=gp3
```

## Ssh into machine and run job

```
ssh elastic_ip
git clone https://github.com/pytorch/data
cd data/benchmarks
python run_benchmark.py
```

Visually inspect logs

## Shut down stack

`aws cloudformation delete-stack --stack-name torchdatabenchmark`
