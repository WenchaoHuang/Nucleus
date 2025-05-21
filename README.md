# Nucleus: C++ Bindings for CUDA Runtime API

## Features
Coming soon...

## Installation
Coming soon...

## Documentation
```cpp
__global__ void add(ns::devPtr<int> out, ns::devPtr<const int> x, ns::devPtr<const int> y, unsigned int count)
{
	NS_FOR(i, count);		out[i] = x[i] + y[i];
}

...

auto context = ns::Context::getInstance();
auto device = context->getDevice(0);
auto stream = device->getDefaultStream();
auto allocator = device->getDefaultAllocator();

int count = 10000;
ns::Array<int> A(allocator, count);
ns::Array<int> B(allocator, count);
ns::Array<int> C(allocator, count);

stream->memset(A.data(), 1, A.size());
stream->memset(B.data(), 2, B.size()).sync();
stream->launch(add, ns::ceil_div(count, 256), 256)(C, A, B, count);
stream->launch(add, ns::ceil_div(count, 256), 256)(A, B, C, count).sync();
stream->memcpy(host_result.data(), A.data(), A.size());
stream->sync();
```

## License
Under the terms of the [MIT License](LICENSE).
