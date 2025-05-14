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
#include "version.hpp"

#include <vector>

namespace NS_NAMESPACE
{
	/*********************************************************************
	***************************    Context    ****************************
	*********************************************************************/

	/**
	 *	@brief		Wrapper for CUDA context object (singleton).
	 */
	class Context
	{
		NS_NONCOPYABLE(Context)

	private:

		//!	@brief		Create CUDA context wrapper.
		Context();

		//!	@brief		Destroy CUDA context wrapper.
		~Context();

	public:

		/**
		 *	@brief		Return a raw pointer to the CUDA context wrapper (singleton).
		 */
		static Context * getInstance()
		{
			static Context s_instance;

			return &s_instance;
		}

	public:

		/**
		 *	@brief		Return the last error from a runtime call.
		 *	@note		Return the last error that has been produced by any of the runtime calls
		 *				in the same host thread and reset it to Error::eSuccess.
		 */
		static cudaError_t getLastError() noexcept;


		/**
		 *	@brief		Return a string containing the name of an error code in the enum.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		static const char * getErrorName(cudaError_t eValue) noexcept;


		/**
		 *	@brief		Return the description string for an error code.
		 *	@note		If the error code is not recognized, "unrecognized error code" is returned.
		 */
		static const char * getErrorString(cudaError_t eValue) noexcept;

	public:

		/**
		 *	@brief		Return the latest version of CUDA supported by the driver.
		 */
		Version getDriverVersion() const { return m_driverVersion; }


		/**
		 *	@brief		Return the version number of the current CUDA Runtime instance.
		 */
		Version getRuntimeVersion() const { return m_runtimeVersion; }


		/**
		 *	@brief		Return pointer to physical device.
		 */
		Device * getDevice(size_t index) const { return m_pNvidiaDevices[index]; }


		/**
		 *	@brief		Return physical device array.
		 */
		const std::vector<Device*> & getDevices() const { return m_pNvidiaDevices; }

	private:

		Version						m_driverVersion;
		Version						m_runtimeVersion;
		std::vector<Device*>		m_pNvidiaDevices;
	};
}