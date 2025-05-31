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

#include "stream.h"
#include <device_launch_parameters.h>

#ifndef __CUDACC__
	// This is a no-op for actual compilation,
	// only for suppressing MSVC IntelliSense errors.
	#define __CUDACC__
	#error This file should be inclued in *.cu files only!
#endif

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    utils    *****************************
	*********************************************************************/

	/**
	 *	@brief		Computes the ceiling of integer division (x/y)
	 *	@example	stream.launch(kernel, ns::ceil_div(count, 256), 256)(...)
	 *	@note		Safe for unsigned integer arithmetic.
	 */
	constexpr uint32_t ceil_div(size_t x, size_t y) { return static_cast<uint32_t>((x + y - 1) / y); }


	/**
	 *	@brief		Returns the globally unique thread ID in a 1D grid-block layout.
	 */
	__device__ __forceinline__ unsigned int tid() { return blockDim.x * blockIdx.x + threadIdx.x; }


	/**
	 *	@brief		Macro that automatically bounds-checks thread ID.
	 *	@details	1. Calculating the global thread ID (i)
	 *				2. Early-exiting if thread ID is out of bounds
	 * 
	 *	@note		- UPPER_CASE (NS_BOUNDS_CHECK): For codebases enforcing strict conventions
	 *				- lower_case (ns_bounds_check): For ergonomic coding and rapid prototyping
	 *				(Can you guess which one I prefer?)
	 */
	#define ns_bounds_check(i, num_threads)		const auto i = ns::tid();	if (i >= (num_threads)) return;
	#define NS_BOUNDS_CHECK(i, num_threads)		ns_bounds_check(i, num_threads)
	#define CUDA_for(i, num_threads)			ns_bounds_check(i, num_threads)

	/*********************************************************************
	****************************    kernel    ****************************
	*********************************************************************/

	namespace kernel
	{
		template<typename Type> __global__ void memset(Type * pValues, Type value, size_t count)
		{
			CUDA_for(i, count);					pValues[i] = value;
		}
	}

	/*********************************************************************
	****************************    Stream    ****************************
	*********************************************************************/

	/**
	 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
	 *	@param[in]	func - Device function symbol.
	 *	@param[in]	gDim - Grid dimensions.
	 *	@param[in]	bDim - Block dimensions.
	 *	@param[in]	sharedMem - Number of bytes for shared memory.
	 *	@example	stream.launch(KernelAdd, gridDim, blockDim, sharedMem)(A, B, C, count);
	 *	@note		The returned lambda is a temporary object that should be used immediately to configure and launch the kernel.
	 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments.
	 *	@warning	Only available in *.cu files (implemented in launch_utils.cuh).
	 */
	template<typename... Args> auto Stream::launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if _HAS_CXX20
		return [=, this](Args... args) -> Stream& { void * params[] = { &args... };		return this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#else
		return [=](Args... args) -> Stream& { void * params[] = { &args... };	return this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#endif
	}

	//	Specialization for parameterless kernels
	template<> inline auto Stream::launch(KernelFunc<> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if _HAS_CXX20
		return [=, this]() -> Stream& { return this->launchKernel(func, gridDim, blockDim, sharedMem, nullptr); };
	#else
		return [=]() -> Stream& { return this->launchKernel(func, gridDim, blockDim, sharedMem, nullptr); };
	#endif
	}


	/**
	 *	@brief		Initialize or set device memory to a value.
	 *	@param[in]	pValues - Pointer to the device memory.
	 *	@param[in]	value - Value to set for.
	 *	@param[in]	count - Count of values to set.
	 *	@param[in]	blockSize - CUDA thread block size (default = 256, which is near-optimal for most modern GPUs).
	 *	@retval		Stream - Reference to this stream (enables method chaining).
	 *	@warning	Only available in *.cu files (CUDA compilation required).
	 */
	template<typename Type> Stream & Stream::memset(Type * pValues, Type value, size_t count, int blockSize)
	{
		return this->launch(kernel::memset<Type>, ns::ceil_div(count, blockSize), blockSize)(pValues, value, count);
	}
}