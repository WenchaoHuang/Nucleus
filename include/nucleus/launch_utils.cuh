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

#include "graph.h"
#include "stream.h"
#include "logger.h"
#include <device_launch_parameters.h>

#ifndef __CUDACC__
	// This is a no-op for actual compilation,
	// only for suppressing MSVC IntelliSense errors.
	#define __CUDACC__
	#error This file should be inclued in *.cu files only!
#endif

namespace NS_NAMESPACE
{
	/*****************************************************************************
	********************************    utils    *********************************
	*****************************************************************************/

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
	#define NS_BOUNDS_CHECK(i, num_threads)		const auto i = ns::tid();	if (i >= (num_threads)) return;
	#define ns_bounds_check(i, num_threads)		NS_BOUNDS_CHECK(i, num_threads)
	#define CUDA_for(i, num_threads)			NS_BOUNDS_CHECK(i, num_threads)

	/*****************************************************************************
	********************************    kernel    ********************************
	*****************************************************************************/

	namespace kernel
	{
		template<typename Type> __global__ void memset(Type * pValues, Type value, size_t count)
		{
			CUDA_for(i, count);					pValues[i] = value;
		}
	}

	/*****************************************************************************
	********************************    Stream    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
	 *	@param[in]	func - Device function symbol.
	 *	@param[in]	gridDim - Grid dimensions.
	 *	@param[in]	blockDim - Block dimensions.
	 *	@param[in]	sharedMem - Number of bytes for shared memory.
	 *	@example	stream.launch(KernelAdd, gridDim, blockDim, sharedMem)(A, B, C, count);
	 *	@note		The returned lambda is a temporary object that should be used immediately to configure and launch the kernel.
	 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments.
	 *	@warning	Only available in *.cu files (implemented in launch_utils.cuh).
	 */
	template<typename... Args> auto Stream::launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if NS_HAS_CXX_20
		return [=, this](Args... args) -> Stream& { void * params[] = { &args... };		return this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#else
		return [=](Args... args) -> Stream& { void * params[] = { &args... };	return this->launchKernel(func, gridDim, blockDim, sharedMem, params); };
	#endif
	}

	//	Specialization for parameterless kernels
	template<> inline auto Stream::launch(KernelFunc<> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem)
	{
	#if NS_HAS_CXX_20
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

	/*****************************************************************************
	********************************    Graph    *********************************
	*****************************************************************************/

	template<typename Type> ExecDep Graph::memset(Type * pValues, Type value, size_t count, std::initializer_list<ExecDep> dependencies)
	{
		constexpr unsigned int [[maybe_unused]] optimal_block_size_RTX_3080_Ti = 512;
		constexpr unsigned int [[maybe_unused]] optimal_block_size_RTX_2070_SUPER = 256;

		constexpr int blockSize = optimal_block_size_RTX_3080_Ti;

		return this->launch(kernel::memset<Type>, dependencies, ns::ceil_div(count, blockSize), blockSize)(pValues, value, count);
	}

	template<typename... Args> ExecDep Graph::launchKernel(KernelFunc<Args...> func, std::initializer_list<ExecDep> dependencies, dim3 gridDim, dim3 blockDim, unsigned int sharedMem, Args... args)
	{
		if (m_pImmediateLaunchStream != nullptr)	//	in immediate launch mode
		{
			m_pImmediateLaunchStream->launch(func, gridDim, blockDim, sharedMem)(args...);

			return ExecDep(m_ID, -1);
		}
		else if (gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z == 0)
		{
			return this->barrier(dependencies);
		}
		else /////////////////////////////////////////////////////////////////////////////////////////////////////////////
		{
			const uint64_t depHash = this->cacheDependencies(dependencies);

			constexpr size_t paramBytes = sizeof(gridDim) + sizeof(blockDim) + sizeof(sharedMem) + (sizeof(args) + ...);

			char paramCache[paramBytes];												size_t paramOffset = 0;
			((std::memcpy(paramCache + paramOffset, &gridDim, sizeof(gridDim)), paramOffset += sizeof(gridDim)));
			((std::memcpy(paramCache + paramOffset, &blockDim, sizeof(blockDim)), paramOffset += sizeof(blockDim)));
			((std::memcpy(paramCache + paramOffset, &sharedMem, sizeof(sharedMem)), paramOffset += sizeof(sharedMem)));
			((std::memcpy(paramCache + paramOffset, &args, sizeof(args)), paramOffset += sizeof(args)), ...);

			if (m_indicator < m_nodes.size())	//	in validating state
			{
				if ((m_nodes[m_indicator].func != (void*)func) || (m_nodes[m_indicator].depHash != depHash))	//	dependencies changes
				{
					m_nodes.resize(m_indicator);
				}
				else if (std::memcmp(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes) != 0)	//	parameters changes
				{
					cudaKernelNodeParams			launchParams = {};
					launchParams.func = func;
					launchParams.extra = nullptr;
					launchParams.kernelParams = nullptr;
					launchParams.sharedMemBytes = sharedMem;
					launchParams.blockDim = blockDim;
					launchParams.gridDim = gridDim;

					if constexpr (sizeof...(Args))
					{
						void * params[] = { ((void*)&args)... };

						launchParams.kernelParams = params;

						cudaError_t err = cudaGraphKernelNodeSetParams(m_nodes[m_indicator].hGraphNode, &launchParams);

						NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));
					}
					else
					{
						cudaError_t err = cudaGraphKernelNodeSetParams(m_nodes[m_indicator].hGraphNode, &launchParams);

						NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));
					}

					std::memcpy(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes);

					m_isParamChg = true;
				}
			}

			if (m_indicator >= m_nodes.size())	//	topology changed
			{
				auto createFunc = [=](cudaGraph_t hGraph, const cudaGraphNode_t * pDependencies, size_t numDependencies) -> cudaGraphNode_t
				{
					cudaGraphNode_t					hGraphNode = nullptr;
					cudaKernelNodeParams			launchParams = {};
					launchParams.func = func;
					launchParams.extra = nullptr;
					launchParams.kernelParams = nullptr;
					launchParams.sharedMemBytes = sharedMem;
					launchParams.blockDim = blockDim;
					launchParams.gridDim = gridDim;

					if constexpr (sizeof...(Args))
					{
						void * params[] = { ((void*)&args)... };	//	where magic happen, all parameters will captured in the lambda!

						launchParams.kernelParams = params;

						cudaError_t err = cudaGraphAddKernelNode(&hGraphNode, hGraph, pDependencies, numDependencies, &launchParams);

						NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));
					}
					else
					{
						cudaError_t err = cudaGraphAddKernelNode(&hGraphNode, hGraph, pDependencies, numDependencies, &launchParams);

						NS_ERROR_LOG_IF(err != cudaSuccess, "%s.", cudaGetErrorString(cudaGetLastError()));
					}

					return hGraphNode;
				};

				m_paramBinaries.resize(m_paramBinaries.size() + paramBytes);

				std::memcpy(m_paramBinaries.data() + m_paramOffset, paramCache, paramBytes);

				m_nodes.emplace_back(func, depHash, paramBytes, m_depIndicesCache, createFunc);

				m_isTopoChg = true;
			}

			m_paramOffset += paramBytes;

			return ExecDep(m_ID, m_indicator++);
		}
	}
}