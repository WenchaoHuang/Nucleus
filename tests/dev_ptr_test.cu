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

#include <nucleus/dev_ptr.h>
#include <device_launch_parameters.h>

/*************************************************************************
***************************    dev_ptr_test    ***************************
*************************************************************************/

__global__ void test(ns::DevPtr<float> out, ns::DevPtr<const float> in, unsigned int num)
{
	auto i = blockDim.x * blockIdx.x + threadIdx.x;

	out.size();
	out.empty();
	out.bytes();
	out.width();
	out.pitch();
	out.data();

	out[i] = in[i];
}


void test_dev_ptr()
{
	ns::DevPtr<float> devPtr0;
	ns::DevPtr<float> devPtr1 = nullptr;
	ns::DevPtr<float> devPtr2(nullptr, 1024);

	ns::DevPtr2<int> devPtr3;
	ns::DevPtr2<int> devPtr4 = nullptr;
	ns::DevPtr2<int> devPtr5(nullptr, 100, 200);

	ns::DevPtr3<bool> devPtr6;
	ns::DevPtr3<bool> devPtr7 = nullptr;
	ns::DevPtr3<bool> devPtr8(nullptr, 100, 200, 300);

	if (devPtr0)
	{
		devPtr0.size();
		devPtr0.empty();
		devPtr0.bytes();
		devPtr0.width();
		devPtr0.pitch();
		devPtr0.data();
		devPtr0 = nullptr;
	}

	if (devPtr3 == devPtr4)
	{
		devPtr4.size();
		devPtr4.empty();
		devPtr4.bytes();
		devPtr4.width();
		devPtr4.height();
		devPtr4.pitch();
		devPtr4.pitch();
		devPtr4.data();
		devPtr4 = nullptr;

	//	ns::DevPtr<int> devPtr = devPtr4;
	}

	if (devPtr8)
	{
		devPtr8.size();
		devPtr8.empty();
		devPtr8.bytes();
		devPtr8.width();
		devPtr8.height();
		devPtr8.pitch();
		devPtr8.pitch();
		devPtr8.pitch();
		devPtr8.data();
		devPtr8 = nullptr;
	}
}