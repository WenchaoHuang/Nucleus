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

		explicit Device(int deviceID);

		~Device();

	public:

		void setCurrent();



	private:

		const int		m_deviceID;
	};
}