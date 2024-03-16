# rk3588-npu
Reverse engineering the rk3588 npu

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
