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

#include <nucleus/device.hpp>
#include <nucleus/context.hpp>
#include <nucleus/array_1d.hpp>
#include <nucleus/array_2d.hpp>
#include <nucleus/array_3d.hpp>

/*************************************************************************
****************************    test_array    ****************************
*************************************************************************/

void test_array()
{
	auto device = ns::Context::getInstance()->getDevice(0);
	auto allocator = device->getDefaultAllocator();

	ns::Array1D<int>	array0;
	ns::Array1D<int>	array1(allocator, 100);
	ns::Array1D<int>	array22 = std::move(array1);

	ns::Array2D<float>	array2;
	ns::Array2D<float>	array3(allocator, 100, 100);

	ns::Array3D<float>	array4;
	ns::Array3D<float>	array5(allocator, 100, 100, 100);

	if (array1.empty())
	{
		int * data = &array1[0];

		array1.size();
		array1.bytes();
		array1.getAllocator();
		array1.reshape(allocator, 200);
		array1.clear();
		array1.data();
	}

	if (array3.empty())
	{
		float * data = &array3[0][2];

		array3.size();
		array3.bytes();
		array3.width();
		array3.pitch();
		array3.height();
		array3.getAllocator();
		array3.reshape(allocator, 200, 400);
		array3.clear();
		array3.data();
	}

	if (array5.empty())
	{
		float * data = &array5[0][0][0];

		array5.size();
		array5.bytes();
		array5.width();
		array5.pitch();
		array5.depth();
		array5.height();
		array5.getAllocator();
		array5.reshape(allocator, 200, 400, 500);
		array5.clear();
		array5.data();
	}
}