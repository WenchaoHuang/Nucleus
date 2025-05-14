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

#include "device.hpp"
#include "logger.hpp"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*************************************************************************
******************************    Device    ******************************
*************************************************************************/
Device::Device(int deviceID, const cudaDeviceProp & devProp) : m_deviceID(deviceID), m_devProp(std::make_unique<cudaDeviceProp>(devProp))
{

}


cudaError_t Device::init() noexcept
{
	this->setCurrent();

	cudaError_t err = cudaFree(nullptr);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return err;
}


size_t Device::getFreeMemorySize() const
{
	this->setCurrent();

	size_t freeMemInBytes = 0, totalMemInBytes = 0;

	cudaError_t err = cudaMemGetInfo(&freeMemInBytes, &totalMemInBytes);

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}

	return freeMemInBytes;
}


void Device::sync() const
{
	this->setCurrent();

	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
	{
		NS_ERROR_LOG("%s.", cudaGetErrorString(err));

		cudaGetLastError();
	}
}


void Device::setCurrent() const
{
	thread_local int currentDevice = 0;

	if (this->m_deviceID != currentDevice)
	{
		cudaError_t err = cudaSetDevice(this->m_deviceID);

		if (err != cudaSuccess)
		{
			NS_ERROR_LOG("%s.", cudaGetErrorString(err));

			cudaGetLastError();
		}
		else
		{
			currentDevice = this->m_deviceID;
		}
	}
}


Device::~Device() noexcept
{

}