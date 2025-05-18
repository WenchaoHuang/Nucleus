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
#pragma once

#include "stream.hpp"
#include <device_launch_parameters.h>

#ifndef __CUDACC__
	#error This file should be inclued in *.cu files only!
#endif

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    kernel    ****************************
	*********************************************************************/

	namespace kernel
	{
		template<typename Type> __global__ void memset(Type * pValues, Type value, size_t count)
		{
			auto tid = blockDim.x * blockIdx.x + threadIdx.x;

			if (tid < count) { pValues[tid] = value; }
		}
	}

	/*********************************************************************
	****************************    Stream    ****************************
	*********************************************************************/

	template<typename... Args> auto Stream::launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if _HAS_CXX20
		return [=, this](Args... args) { void * params[] = { &args... };	this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#else
		return [=](Args... args) { void * params[] = { &args... };	this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#endif
	}

	template<> inline auto Stream::launch(KernelFunc<> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if _HAS_CXX20
		return [=, this]() { this->launchKernel(func, gridDim, blockDim, sharedMem, nullptr); };
	#else
		return [=]() { this->launchKernel(func, gridDim, blockDim, sharedMem, nullptr); };
	#endif
	}

	template<typename Type> Stream & Stream::memset(Type * pValues, Type value, size_t count, int blockSize)
	{
		this->launch(kernel::memset<Type>, ((int)count + blockSize - 1) / blockSize, blockSize)(pValues, value, count);
	}
}