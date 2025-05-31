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

#include "event.h"
#include "logger.h"
#include "device.h"
#include "stream.h"
#include <cuda_runtime.h>

NS_USING_NAMESPACE

/*************************************************************************
******************************    Stream    ******************************
*************************************************************************/
Stream::Stream(Device * device, int priority) : m_device(device), m_hStream(nullptr), m_priority(priority)
{
	device->setCurrent();

	//auto priorityRange = device->getStreamPriorityRange();

	//priority = Math::Clamp(priority, priorityRange.least, priorityRange.greatest);

	cudaError_t err = cudaStreamCreateWithPriority(&m_hStream, cudaStreamNonBlocking, -priority);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


Stream::Stream(Device * device, std::nullptr_t) : m_device(device), m_hStream(nullptr), m_priority(0)
{

}


Stream & Stream::recordEvent(Event & event)
{
	this->acquireDeviceContext();

	cudaError_t err = cudaEventRecord(event.getHandle(), m_hStream);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::waitEvent(Event & event)
{
	this->acquireDeviceContext();

	cudaError_t err = cudaStreamWaitEvent(m_hStream, event.getHandle(), cudaEventWaitDefault);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::launchGraph(cudaGraphExec_t hGraphExec)
{
	this->acquireDeviceContext();

	cudaError_t err = cudaGraphLaunch(hGraphExec, m_hStream);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::launchHostFunc(HostFunc<void> func, void * userData)
{
	this->acquireDeviceContext();

	cudaError_t err = cudaLaunchHostFunc(m_hStream, func, userData);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


Stream & Stream::launchKernel(const void * func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem, void ** args)
{
	if (gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z != 0)
	{
		this->acquireDeviceContext();

		cudaError_t err = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, m_hStream);

	#ifdef NS_CUDA_MEMORY_CHECK
		cudaStreamSynchronize(m_hStream);
	#endif

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();

			throw err;
		}
	}

	return *this;
}


void Stream::memcpy3D(void * dst, size_t dstPitch, size_t dstHeight, const void * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth)
{
	this->acquireDeviceContext();

	cudaMemcpy3DParms mempcyParams = {};
	mempcyParams.dstPtr = make_cudaPitchedPtr(dst, dstPitch, 0, dstHeight);
	mempcyParams.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), srcPitch, 0, srcHeight);
	mempcyParams.extent = make_cudaExtent(width, height, depth);
	mempcyParams.kind = cudaMemcpyDefault;

	cudaError_t err = cudaMemcpy3DAsync(&mempcyParams, m_hStream);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


Stream & Stream::memsetZero(void * address, size_t bytes)
{
	this->acquireDeviceContext();

	cudaError_t err = cudaMemsetAsync(address, 0, bytes, m_hStream);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return *this;
}


void Stream::sync() const
{
	this->acquireDeviceContext();

	cudaError_t err = cudaStreamSynchronize(m_hStream);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


bool Stream::query() const
{
	this->acquireDeviceContext();

	cudaError_t err = cudaStreamQuery(m_hStream);

	if ((err != cudaSuccess) && (err != cudaErrorNotReady))
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err == cudaSuccess;
}


void Stream::acquireDeviceContext() const
{
	if (m_hStream == nullptr)
	{
		m_device->setCurrent();
	}
}


Stream::~Stream() noexcept
{
	if (m_hStream != nullptr)
	{
		cudaError_t err = cudaStreamDestroy(m_hStream);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}