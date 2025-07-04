# Nucleus: C++ Bindings for CUDA
**This project is in active development.** The API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

## Overview
Nucleus is an open-source CUDA C++ API designed to streamline heterogeneous computing development. With a focus on modularity and maintainability, Nucleus provides modern C++ abstractions for CUDA programming, making your code safer, clearer, and easier to integrate.

## Features
- **CUDA Header Isolation**  
  Nucleus isolates CUDA headers, reducing conflicts and simplifying integration with different C++ toolchains and projects.
- **Extensive Compile-Time Type Checking**  
  Enjoy robust compile-time type checks to catch errors early and ensure code safety without sacrificing performance.
- **Unique Runtime Memory Checking Mechanism**  
  Nucleus provides a unique runtime memory checking mechanism to help detect and prevent memory errors during execution.
- **Well-Structured Code**  
  Nucleus emphasizes clean, well-structured, and maintainable code for a better development experience.

## Installation

### Prerequisites

- CUDA Toolkit (version 11.0 or above)
- CMake (version 3.18 or above)
- C++17 compatible compiler

### Build Instructions

```bash
git clone https://github.com/WenchaoHuang/Nucleus.git
cd Nucleus
mkdir build && cd build
cmake ..
```

## Documentation
```cpp
// Example CUDA kernel
__global__ void add_kernel(dev::Ptr<int> out, dev::Ptr<const int> x, dev::Ptr<const int> y, unsigned int count)
{
	CUDA_for(i, count);

	out[i] = x[i] + y[i];
}

...

auto context = ns::Context::getInstance();
auto device = context->getDevice(0);
auto & stream = device->getDefaultStream();
auto allocator = device->getDefaultAllocator();

int count = 10000;
ns::Array<int> A(allocator, count);
ns::Array<int> B(allocator, count);
ns::Array<int> C(allocator, count);

stream.memset(A.data(), 1, A.size());
stream.memset(B.data(), 2, B.size()).sync();
stream.launch(add_kernel, ns::ceil_div(count, 256), 256)(C, A, B, count);
stream.launch(add_kernel, ns::ceil_div(count, 256), 256)(A, B, C, count).sync();
stream.memcpy(host_result.data(), A.data(), A.size());
stream.sync();
```

## License
Nucleus is distributed under the terms of the [MIT License](LICENSE).
