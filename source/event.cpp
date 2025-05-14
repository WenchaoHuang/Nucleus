/**
 *	Copyright (c) 2025 Huang Wenchao <physhuangwenchao@gmail.com>
 *
 *	All rights reserved. Use of this source code is governed by a
 *	GPL-2.0 license that can be found in the LICENSE file.
 *
 *	This program is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *	GNU General Public License for more details.
 */

#include "event.hpp"
#include "device.hpp"
#include "logger.hpp"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*************************************************************************
******************************    Event    *******************************
*************************************************************************/
Event::Event(class Device * pDevice, bool isBlockingSync, bool isDisableTiming) : m_pDevice(pDevice), m_hEvent(nullptr), m_isBlockingSync(isBlockingSync), m_isDisableTiming(isDisableTiming)
{
	NS_ASSERT(pDevice != nullptr);

	pDevice->setCurrent();

	unsigned int flags = cudaEventDefault;

	if (isBlockingSync)		flags |= cudaEventBlockingSync;
	if (isDisableTiming)	flags |= cudaEventDisableTiming;

	cudaError_t err = cudaEventCreateWithFlags(const_cast<cudaEvent_t*>(&m_hEvent), flags);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();

		throw err;
	}
}


std::chrono::nanoseconds Event::getElapsedTime(cudaEvent_t hEventStart, cudaEvent_t hEventEnd)
{
	float elapsedTime = 0.0f;

	cudaError_t err = cudaEventElapsedTime(&elapsedTime, hEventStart, hEventEnd);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return std::chrono::nanoseconds(static_cast<long long>(elapsedTime * 1e6));
}


void Event::sync() const
{
	cudaError_t err = cudaEventSynchronize(m_hEvent);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


bool Event::query() const
{
	cudaError_t err = cudaEventQuery(m_hEvent);

	if ((err != cudaSuccess) && (err != cudaErrorNotReady))
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err == cudaSuccess;
}


Event::~Event() noexcept
{
	if (m_hEvent != nullptr)
	{
		cudaError_t err = cudaEventDestroy(m_hEvent);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
	}
}