# Orthrus

## Development

### Dependencies

- [CMake >=3.18](https://cmake.org/)
- [GCC >=12](https://gcc.gnu.org/) (C++20 support required)
- [OpenSSL](https://www.openssl.org/)
- [Protobuf](https://developers.google.com/protocol-buffers)
- [Google Logging Library](https://github.com/google/glog)

For building the CUDA version, you will also need:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

Make sure your `nvcc` is compatible with your GCC version.

### Building

```bash
mkdir build
cd build
cmake ..
make -j`nproc`
```

Please adjust according to your setup as needed. Tested only on Ubuntu 22.04
and later. This program requires C++20 support, and makes use of some
Linux-specific system calls (like `memfd_create`), although in a limited way.

For more information on building this project, please take a look at the
[Dockerfile.amd64](docker/Dockerfile.amd64) and
[Dockerfile.cuda](docker/Dockerfile.cuda) files.

### Test Models

For testing purposes, you can use
[tinyllamas](https://huggingface.co/karpathy/tinyllamas). Please use the
`tools/bin2glint.py` script to convert `.bin` files to Orthrus's format. There
are other scripts in the `tools` directory for converting the original Llama2
models.
