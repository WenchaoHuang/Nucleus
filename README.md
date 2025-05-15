# Nucleus: C++ Bindings for CUDA Runtime API

## Features
Coming soon...

## Installation
Coming soon...

## Documentation
```
	auto context = ns::Context::getInstance();
	auto driverVersion = context->getDriverVersion();
	auto runtimVersion = context->getRuntimeVersion();
	auto devices = context->getDevices();
	auto device = context->getDevice(0);

	device->init();
	device->getDeviceProperties();
	device->getFreeMemorySize();
	device->sync();
```

## License
Under the terms of the [MIT License](LICENSE).
