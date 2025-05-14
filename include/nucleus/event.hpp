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
#include <chrono>

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Event    *****************************
	*********************************************************************/

	/**
	 *	@brief		RAII wrapper for CUDA event object.
	 */
	class Event
	{
		NS_NONCOPYABLE(Event)

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	pDevice - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@param[in]	isDisableTiming - Specify that the created event does not need to record timing data.
		 *	@note		If isBlockingSync is set, thread calling Event::Synchronize() will give up CPU time until event happened (default method).
		 *	@note		If isBlockingSync is not set, thread calling Event::Synchronize() will enter a check-event loop until event happened, results in the minimum latency.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Event(Device * pDevice, bool isBlockingSync = false, bool isDisableTiming = false);


		/**
		 *	@brief		Destroy CUDA event object.
		 */
		~Event() noexcept;

	public:

		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * getDevice() const { return m_pDevice; }


		/**
		 *	@brief		Return CUDA event type of this object.
		 *	@warning	Only for CUDA-based project use.
		 */
		cudaEvent_t getHandle() noexcept { return m_hEvent; }


		/**
		 *	@brief		Compute the elapsed time between events.
		 *	@param[in]	hEventStart - Valid starting event handle.
		 *	@param[in]	hEventEnd - Valid ending event handle.
		 *	@warning	pEventStart and pEventEnd must from the same device.
		 *	@note		With a resolution of around 0.5 microseconds.
		 */
		static std::chrono::nanoseconds getElapsedTime(cudaEvent_t hEventStart, cudaEvent_t hEventEnd);


		/**
		 *	@brief		Wait for an event to complete.
		 *	@note		Wait until the completion of all work currently captured in this event.
		 */
		void sync() const;


		/**
		 *	@brief		Query an event's status.
		 *	@retval		True - If all captured work has been completed.
		 */
		bool query() const;

	private:

		const cudaEvent_t 			m_hEvent;
		Device * const				m_pDevice;
		const bool					m_isBlockingSync;
		const bool					m_isDisableTiming;
	};
}