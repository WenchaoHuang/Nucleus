/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */

#include <nucleus/event.h>
#include <nucleus/stream.h>
#include <nucleus/context.h>
#include <nucleus/allocator.h>
#include <nucleus/scoped_device.h>

/*********************************************************************************
****************************    shared_handle_test    ****************************
*********************************************************************************/

void scoped_device_test()
{
	auto & devices = ns::Context::getInstance()->getDevices();
	auto device0 = *devices.begin();
	auto device1 = *devices.rbegin();

	assert(ns::ScopedDevice::getCurrent() == device0);
	{
		ns::ScopedDevice	scope(device1);

		assert(ns::ScopedDevice::getCurrent() == device1);

		ns::Event				event;
		ns::Stream				stream;
		ns::DeviceAllocator		allocator;

		ns::SharedEvent		xxx;

		assert(event.device() == device1);
		assert(stream.device() == device1);
		assert(allocator.device() == device1);
	}
	assert(ns::ScopedDevice::getCurrent() == device0);
}