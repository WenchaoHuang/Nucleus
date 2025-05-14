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

#include "logger.hpp"
#include "device.hpp"
#include "context.hpp"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

/*************************************************************************
*****************************    Context    ******************************
*************************************************************************/
Context::Context()
{
	cudaGetLastError();

	//////////////////////////////////////////////////////////////////////

	int driverVersion = 0;

	cudaDriverGetVersion(&driverVersion);

	m_driverVersion.Major = driverVersion / 1000;
	m_driverVersion.Minor = (driverVersion % 1000) / 10;

	NS_INFO_LOG("CUDA driver version: %d.%d", m_driverVersion.Major, m_driverVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	int runtimeVersion = 0;

	cudaRuntimeGetVersion(&runtimeVersion);

	m_runtimeVersion.Major = runtimeVersion / 1000;
	m_runtimeVersion.Minor = (runtimeVersion % 1000) / 10;

	NS_INFO_LOG("CUDA runtime version: %d.%d", m_runtimeVersion.Major, m_runtimeVersion.Minor);

	//////////////////////////////////////////////////////////////////////

	cudaGetLastError();

	int deviceCount = 0;

	auto err = cudaGetDeviceCount(&deviceCount);

	m_pNvidiaDevices.resize(deviceCount, nullptr);

	NS_INFO_LOG_IF(err == cudaErrorNoDevice, "No CUDA-capable devices were detected.");

	//////////////////////////////////////////////////////////////////////

	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp = {};

		cudaGetDeviceProperties(reinterpret_cast<cudaDeviceProp*>(&devProp), i);

		NS_INFO_LOG("CUDA device(%d): %s, compute capability: %d.%d", i, devProp.name, devProp.major, devProp.minor);

		m_pNvidiaDevices[i] = new Device(i, devProp);
	}

	cudaGetLastError();
}


const char * Context::getErrorString(cudaError_t eValue) noexcept
{
	return cudaGetErrorString(eValue);
}


const char * Context::getErrorName(cudaError_t eValue) noexcept
{
	return cudaGetErrorName(eValue);
}


cudaError_t Context::getLastError() noexcept
{
	return cudaGetLastError();
}


Context::~Context()
{
	for (size_t i = 0; i < m_pNvidiaDevices.size(); i++)
	{
		delete m_pNvidiaDevices[i];
	}

	m_pNvidiaDevices.clear();
}