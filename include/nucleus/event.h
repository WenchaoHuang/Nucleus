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

	private:

		friend class TimedEvent;

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@param[in]	isDisableTiming - Specify that the created event does not need to record timing data.
		 *	@details	If isBlockingSync is set, thread calling Event::sync() will give up CPU time until event happened (default method).
		 *	@details	If isBlockingSync is not set, thread calling Event::sync() will enter a check-event loop until event happened, results in the minimum latency.
		 *	@throw		cudaError_t - In case of failure.
		 *	@note		Designed for class `TimedEvent`.
		 */
		explicit Event(Device * device, bool isBlockingSync, bool isDisableTiming);

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@details	If isBlockingSync is set, thread calling Event::sync() will give up CPU time until event happened (default method).
		 *	@details	If isBlockingSync is not set, thread calling Event::sync() will enter a check-event loop until event happened, results in the minimum latency.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Event(Device * device, bool isBlockingSync = false) : Event(device, isBlockingSync, true) {}


		/**
		 *	@brief		Destroy CUDA event object.
		 */
		~Event() noexcept;

	public:

		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * getDevice() const { return m_device; }


		/**
		 *	@brief		Return CUDA event type of this object.
		 *	@warning	Only for CUDA-based project use.
		 */
		cudaEvent_t getHandle() noexcept { return m_hEvent; }


		/**
		 *	@brief		Query an event's status.
		 *	@retval		True - If all captured work has been completed.
		 */
		bool query() const;


		/**
		 *	@brief		Wait for an event to complete.
		 *	@note		Wait until the completion of all work currently captured in this event.
		 */
		void sync() const;

	private:

		cudaEvent_t				m_hEvent;
		Device * const			m_device;
		const bool				m_isBlockingSync;
	};

	/*********************************************************************
	**************************    TimedEvent    **************************
	*********************************************************************/

	/**
	 *	@brief		RAII wrapper for timed CUDA event object.
	 */
	class TimedEvent : public Event
	{

	public:

		/**
		 *	@brief		Create CUDA event object.
		 *	@param[in]	device - Pointer to the device associated with.
		 *	@param[in]	isBlockingSync - Specify that event should use blocking synchronization.
		 *	@details	If isBlockingSync is set, thread calling Event::sync() will give up CPU time until event happened (default method).
		 *	@details	If isBlockingSync is not set, thread calling Event::sync() will enter a check-event loop until event happened, results in the minimum latency.
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit TimedEvent(Device * device, bool isBlockingSync = false) : Event(device, isBlockingSync, false) {}

	public:

		/**
		 *	@brief		Compute the elapsed time between events.
		 *	@param[in]	eventStart - Valid starting event.
		 *	@param[in]	eventEnd - Valid ending event.
		 *	@warning	eventStart and eventEnd must from the same device.
		 *	@note		With a resolution of around 0.5 microseconds.
		 */
		static std::chrono::nanoseconds getElapsedTime(TimedEvent & eventStart, TimedEvent & eventEnd);
	};
}