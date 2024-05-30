# rk3588-npu
Reverse engineering the rk3588 npu.

Internals of the NPU, similarity to NVDLA and matrix muplication support covered in blog [post](http://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html).
Example integration llama2.c running tinystories covered in blog [post](http://jas-hacks.blogspot.com/2024/05/rk3588-reverse-engineering-rknn-running.html)

To build :
```
mkdir build
meson build
cd build
ninja
```

To run tests (tested against 5.10 kernel) :
```
ninja -C build test
```
