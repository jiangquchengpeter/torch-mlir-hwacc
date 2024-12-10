# The Torch-MLIR Project

The Torch-MLIR project aims to provide first class compiler support from the [PyTorch](https://pytorch.org) ecosystem to the MLIR ecosystem.

> This project is participating in the LLVM Incubator process: as such, it is
not part of any official LLVM release.  While incubation status is not
necessarily a reflection of the completeness or stability of the code, it
does indicate that the project is not yet endorsed as a component of LLVM.

[PyTorch](https://pytorch.org)
PyTorch is an open source machine learning framework that facilitates the seamless transition from research and prototyping to production-level deployment.

[MLIR](https://mlir.llvm.org)
The MLIR project offers a novel approach for building extensible and reusable compiler architectures, which address the issue of software fragmentation, reduce the cost of developing domain-specific compilers, improve compilation for heterogeneous hardware, and promote compatibility between existing compilers.

[Torch-MLIR](https://github.com/llvm/torch-mlir)
Several vendors have adopted MLIR as the middle layer in their systems, enabling them to map frameworks such as PyTorch, JAX, and TensorFlow into MLIR and subsequently lower them to their target hardware. We have observed half a dozen custom lowerings from PyTorch to MLIR, making it easier for hardware vendors to focus on their unique value, rather than needing to implement yet another PyTorch frontend for MLIR. The ultimate aim is to be similar to the current hardware vendors adding LLVM target support, rather than each one implementing Clang or a C++ frontend.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


## BUILD CMD
```bash
cmake -GNinja -Bbuild \
  externals/llvm-project/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# use clang`\
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  `# use ccache to cache build results` \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  `# use LLD to link in seconds, rather than minutes` \
  `# if using clang <= 13, replace --ld-path=ld.lld with -fuse-ld=lld` \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="--ld-path=ld.lld" \
  `# Enabling libtorch binary cache instead of downloading the latest libtorch everytime.` \
  `# Testing against a mismatched version of libtorch may cause failures` \
  -DLIBTORCH_CACHE=ON \
  `# Enable an experimental path to build libtorch (and PyTorch wheels) from source,` \
  `# instead of downloading them` \
  -DLIBTORCH_SRC_BUILD=ON \
  `# Set the variant of libtorch to build / link against. (shared|static and optionally cxxabi11)` \
  -DLIBTORCH_VARIANT=shared \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
;
```



## All the roads from PyTorch to Torch MLIR Dialect

We have few paths to lower down to the Torch MLIR Dialect.
 - ONNX as the entry points.
 - Fx as the entry points

## Project Communication

- `#torch-mlir` channel on the LLVM [Discord](https://discord.gg/xS7Z362) - this is the most active communication channel
- Github issues [here](https://github.com/llvm/torch-mlir/issues)
- [`torch-mlir` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/torch-mlir/41) of LLVM Discourse

## Install torch-mlir snapshot

At the time of writing, we release [pre-built snapshots of torch-mlir](https://github.com/llvm/torch-mlir-release) for Python 3.11 and Python 3.10.

If you have supported Python version, the following commands initialize a virtual environment.
```shell
python3.11 -m venv mlir_venv
source mlir_venv/bin/activate
```

Or, if you want to switch over multiple versions of Python using conda, you can create a conda environment with Python 3.11.
```shell
conda create -n torch-mlir python=3.11
conda activate torch-mlir
python -m pip install --upgrade pip
```

Then, we can install torch-mlir with the corresponding torch and torchvision nightlies.
```
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

## Using torch-mlir

Torch-MLIR is primarily a project that is integrated into compilers to bridge them to PyTorch and ONNX. If contemplating a new integration, it may be helpful to refer to existing downstreams:

* [IREE](https://github.com/iree-org/iree.git)
* [Blade](https://github.com/alibaba/BladeDISC)

While most of the project is exercised via testing paths, there are some ways that an end user can directly use the APIs without further integration:

### FxImporter ResNet18
```shell
# Get the latest example if you haven't checked out the code
wget https://raw.githubusercontent.com/llvm/torch-mlir/main/projects/pt1/examples/fximporter_resnet18.py

# Run ResNet18 as a standalone script.
python projects/pt1/examples/fximporter_resnet18.py

# Output
load image from https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg
...
PyTorch prediction
[('Labrador retriever', 70.65674591064453), ('golden retriever', 4.988346099853516), ('Saluki, gazelle hound', 4.477451324462891)]
torch-mlir prediction
[('Labrador retriever', 70.6567153930664), ('golden retriever', 4.988325119018555), ('Saluki, gazelle hound', 4.477458477020264)]
```

## Repository Layout

The project follows the conventions of typical MLIR-based projects:

* `include/torch-mlir`, `lib` structure for C++ MLIR compiler dialects/passes.
* `test` for holding test code.
* `tools` for `torch-mlir-opt` and such.
* `python` top level directory for Python code

## Developers
If you would like to develop and build torch-mlir from source please look at [Development Notes](docs/development.md)
