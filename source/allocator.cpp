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

#include "format.hpp"
#include "logger.hpp"
#include "device.hpp"
#include "allocator.hpp"
#include <cuda_runtime.h>

NS_USING_NAMESPACE

/*************************************************************************
**************************    HostAllocator    ***************************
*************************************************************************/

void * HostAllocator::doAllocateMemory(size_t bytes)
{
	void * ptr = nullptr;

	cudaError_t err = cudaMallocHost(&ptr, bytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef NS_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);
#endif

	return ptr;
}


void HostAllocator::doDeallocateMemory(void * ptr)
{
	cudaError_t err = cudaFreeHost(ptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


/*************************************************************************
*************************    DeviceAllocator    **************************
*************************************************************************/
DeviceAllocator::DeviceAllocator(class Device * pDevice) : m_pDevice(pDevice)
{
	NS_ASSERT(pDevice != nullptr);
}


static cudaChannelFormatDesc CreateChannelDesc(Format eFormat)
{
	cudaChannelFormatDesc channelDesc = {};

	switch (eFormat)
	{
		case Format::eChar:		{ channelDesc = cudaCreateChannelDesc<char1>();		break; }
		case Format::eChar2:	{ channelDesc = cudaCreateChannelDesc<char2>();		break; }
		case Format::eChar4:	{ channelDesc = cudaCreateChannelDesc<char4>();		break; }
		case Format::eUchar:	{ channelDesc = cudaCreateChannelDesc<uchar1>();	break; }
		case Format::eUchar2:	{ channelDesc = cudaCreateChannelDesc<uchar2>();	break; }
		case Format::eUchar4:	{ channelDesc = cudaCreateChannelDesc<uchar4>();	break; }
		case Format::eShort:	{ channelDesc = cudaCreateChannelDesc<short1>();	break; }
		case Format::eShort2:	{ channelDesc = cudaCreateChannelDesc<short2>();	break; }
		case Format::eShort4:	{ channelDesc = cudaCreateChannelDesc<short4>();	break; }
		case Format::eUshort:	{ channelDesc = cudaCreateChannelDesc<ushort1>();	break; }
		case Format::eUshort2:	{ channelDesc = cudaCreateChannelDesc<ushort2>();	break; }
		case Format::eUshort4:	{ channelDesc = cudaCreateChannelDesc<ushort4>();	break; }
		case Format::eInt:		{ channelDesc = cudaCreateChannelDesc<int1>();		break; }
		case Format::eInt2:		{ channelDesc = cudaCreateChannelDesc<int2>();		break; }
		case Format::eInt4:		{ channelDesc = cudaCreateChannelDesc<int4>();		break; }
		case Format::eUint:		{ channelDesc = cudaCreateChannelDesc<uint1>();		break; }
		case Format::eUint2:	{ channelDesc = cudaCreateChannelDesc<uint2>();		break; }
		case Format::eUint4:	{ channelDesc = cudaCreateChannelDesc<uint4>();		break; }
		case Format::eFloat:	{ channelDesc = cudaCreateChannelDesc<float1>();	break; }
		case Format::eFloat2:	{ channelDesc = cudaCreateChannelDesc<float2>();	break; }
		case Format::eFloat4:	{ channelDesc = cudaCreateChannelDesc<float4>();	break; }
		default:				{ NS_ERROR_LOG("Invalid format!");					break; }
	}

	return channelDesc;
}


cudaArray_t DeviceAllocator::allocateTextureMemory(Format eFormat, size_t width, size_t height, size_t depth, int flags)
{
	m_pDevice->setCurrent();

	cudaArray_t hArray = nullptr;

	cudaChannelFormatDesc channelDesc = CreateChannelDesc(eFormat);

	cudaError_t err = cudaMalloc3DArray(&hArray, &channelDesc, make_cudaExtent(width, height, depth), flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return hArray;
}


cudaMipmappedArray_t DeviceAllocator::allocateMipmapTextureMemory(Format eFormat, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags)
{
	m_pDevice->setCurrent();

	cudaMipmappedArray_t hMipmapedArray = nullptr;

	cudaChannelFormatDesc channelDesc = CreateChannelDesc(eFormat);

	cudaError_t err = cudaMallocMipmappedArray(&hMipmapedArray, &channelDesc, make_cudaExtent(width, height, depth), numLevels, flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

	return hMipmapedArray;
}


void DeviceAllocator::deallocateMipmapTextureMemory(cudaMipmappedArray_t hMipmapedArray)
{
	cudaError_t err = cudaFreeMipmappedArray(hMipmapedArray);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void DeviceAllocator::deallocateTextureMemory(cudaArray_t hArray)
{
	cudaError_t err = cudaFreeArray(hArray);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void * DeviceAllocator::doAllocateMemory(size_t bytes)
{
	void * ptr = nullptr;

	m_pDevice->setCurrent();

	cudaError_t err = cudaMalloc(&ptr, bytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}

#ifdef NS_CUDA_MEMORY_CHECK
	err = cudaMemset(ptr, -1, bytes);

	cudaDeviceSynchronize();
#endif

	return ptr;
}


void DeviceAllocator::doDeallocateMemory(void * ptr)
{
	cudaError_t err = cudaFree(ptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}