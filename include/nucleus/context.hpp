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

		Context();

		~Context();

	public:

		static Context * getInstance()
		{
			static Context s_instance;

			return &s_instance;
		}

	public:

		Device * getDevice(size_t index) const { return m_devices[index]; }

		const std::vector<Device*> & getDevices() const { return m_devices; }

		void setCurrentDevice(Device * device);

		Device * getCurrentDevice();




	private:

		std::vector<Device*>		m_devices;
	};
}