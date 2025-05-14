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
#pragma once

#include "fwd.hpp"

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Device    ****************************
	*********************************************************************/

	/**
	 *	@brief		Wrapper for CUDA device object.
	 */
	class Device
	{
		friend class Context;

	private:

		//!	@brief		Create device object.
		Device(int, const cudaDeviceProp&);

		//!	@brief		Destroy device object.
		~Device() noexcept;

	public:


		/**
		 *	@brief		Trigger initialization of the CUDA context.
		 *	@retval		cudaSuccess - If device's context was initialized successfully.
		 */
		cudaError_t init() noexcept;


		/**
		 *	@brief		Wait for compute device to finish.
		 *	@note		Block until the device has completed all preceding requested tasks.
		 */
		void sync() const;


		/**
		 *	@brief		Set device to be used for GPU executions.
		 *	@note		Mainly for internal call of Stream::Handle().
		 *	@note		Set device as the current device for the calling host thread.
		 *	@note		This call may be made from any host thread, to any device, and at
		 *				any time.  This function will do no synchronization with the previous
		 *				or new device, and should be considered a very low overhead call.
		 *	@warning	Callling ::cudaSetDevice() in other place is not allowed!
		 */
		void setCurrent() const;


		/**
		 *	@brief		Query the size of free device memory.
		 *	@return		The free amount of memory available for allocation by the device in bytes.
		 */
		size_t getFreeMemorySize() const;


		/**
		 *	@brief		Return the device properties.
		 *	@note		Requires CUDA Toolkit.
		 */
		const cudaDeviceProp * getDeviceProperties() const { return m_devProp.get(); }

	private:

		const int									m_deviceID;
		const std::unique_ptr<cudaDeviceProp>		m_devProp;
	};
}