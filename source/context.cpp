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
#include "context.hpp"
#include <cuda_runtime_api.h>

NS_USING_NAMESPACE

static thread_local int sgl_deviceID = 0;

/*************************************************************************
*****************************    Context    ******************************
*************************************************************************/
Context::Context()
{


}


void Context::setCurrentDevice(Device * device)
{
	if (device->m_deviceID != sgl_deviceID)
	{
		cudaSetDevice(device->m_deviceID);

		sgl_deviceID = device->m_deviceID;
	}
}


Device * Context::getCurrentDevice()
{
	return m_devices[sgl_deviceID];
}


Context::~Context()
{
	for (size_t i = 0; i < m_devices.size(); i++)
	{
		delete m_devices[i];
	}

	m_devices.clear();
}