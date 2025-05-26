# Nucleus: C++ Bindings for CUDA Runtime API
This project is in active development. This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

## Features
Coming soon...

## Installation
Coming soon...

## Documentation
```cpp
__global__ void add(dev::Ptr<int> out, dev::Ptr<const int> x, dev::Ptr<const int> y, unsigned int count)
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
Nucleus is distributed under the terms of the [MIT License](LICENSE).
