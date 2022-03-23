# Build TorchData Release

These are a collection of scripts that are to be used for release activities.

## Conda

### Release

```bash
PYTHON_VERSION=3.9 PYTORCH_VERSION=1.11.0 packaging/build_conda.sh
```

### Nightly

```bash
PYTHON_VERSION=3.9 packaging/build_conda.sh
```

## Wheel

### Release

```bash
PYTHON_VERSION=3.9 PYTORCH_VERSION=1.11.0 packaging/build_wheel.sh
```

### Nightly

```bash
PYTHON_VERSION=3.9 packaging/build_wheel.sh
```
