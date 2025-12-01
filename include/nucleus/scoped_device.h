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
#pragma once

#include "fwd.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*****************************    ScopedDevice    *****************************
	*****************************************************************************/

	/**
	 *	@brief		A RAII helper class for managing the currently active Device object.
	 * 
	 *	@details	When an instance of ScopedDevice is created, it temporarily sets the
	 *				global current device to the specified one. Upon destruction, the 
	 *				previously active device is automatically restored.
	 * 
	 *	@note		This allows safe and structured switching of devices within a scoped
	 *				context, ensuring proper cleanup and exception safety. Useful when
	 *				operations need to be executed against different devices without
	 *				permanently altering the global state.
	 * 
	 *	@example	{
	 *					ScopedDevice scope(deviceA);					// Current device becomes deviceA
	 *														
	 *					Event				event;						// Create `Event` on deviceA
	 *					Stream				stream;						// Create `Stream` on deviceA
	 *					DeviceAllocator		allocator;					// Create `DeviceAllocator` associated with deviceA
	 * 
	 *				}	// Restores previous device automatically
	 */
	class NS_API ScopedDevice
	{
		NS_NONCOPYABLE(ScopedDevice)

	public:

		/**
		 *	@brief		Creates a scoped device guard.
		 *	@details	Temporarily switches the global active device to the provided one.
		 *	@details	The previous device is stored and will be restored upon destruction.
		 *	@warning	\p device must be a valid pointer to `Device`.
		 */
		explicit ScopedDevice(Device * device) : m_prevDevice(ScopedDevice::getInstance())
		{
			NS_ASSERT(device != nullptr);

			if (device != nullptr)
			{
				ScopedDevice::getInstance() = device;
			}
		}


		/**
		 *	@brief		Destructor restores the previously active device.
		 *	@details	This ensures that device switches are properly nested and exception-safe.
		 */
		~ScopedDevice() noexcept
		{
			ScopedDevice::getInstance() = m_prevDevice;
		}


		/**
		 *	@brief		Returns the currently active device.
		 */
		static Device * getCurrent() { return getInstance(); }

	private:

		using DevicePtr = Device*;

		/**
		 *	@brief		Returns a reference to the static global device pointer.
		 *	@note		Used internally to modify the active device.
		 */
		static DevicePtr & getInstance();

	private:

		Device * const		m_prevDevice;
	};
}