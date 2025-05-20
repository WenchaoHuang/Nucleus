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
	template<typename Type> using HostFunc = void(*)(Type*);

	/*********************************************************************
	****************************    Stream    ****************************
	*********************************************************************/

	/**
	 *	@brief		Wrapper for CUDA stream object.
	 */
	class Stream
	{
		NS_NONCOPYABLE(Stream)

	private:

		friend class Device;

		/**
		 *	@brief		Create a default Stream.
		 *	@param[in]	device - Device associated with.
		 *	@note		Invoked for class Device only.
		 */
		explicit Stream(Device * device, std::nullptr_t);

	public:

		/**
		 *	@brief		Create a non-blocking CUDA Stream.
		 *	@param[in]	device - Device associated with.
		 *	@param[in]	priority - Stream priority.
		 */
		explicit Stream(Device * device, int priority = 0);


		/**
		 *	@brief		Destroy CUDA stream object.
		 */
		~Stream() noexcept;

	public:

		/**
		 *	@brief		Wait for stream tasks to complete.
		 *	@note		Wait until this stream has completed all operations.
		 */
		void sync() const;


		/**
		 *	@brief		query an asynchronous stream for completion status.
		 *	@retval		True - If all operations in this stream have completed.
		 */
		bool query() const;


		/**
		 *	@brief		Return pointer to the device associated with.
		 */
		Device * getDevice() { return m_device; }


		/**
		 *	@brief		Return stream priority.
		 */
		int getPriority() const { return m_priority; }


		/**
		 *	@brief		Return CUDA stream type of this object.
		 *	@warning	Only for CUDA-based project.
		 */
		cudaStream_t getHandle() noexcept { return m_hStream; }

	public:

		/**
		 *	@brief		Record an event.
		 *	@param[in]	hEvent - Valid event handle to record.
		 *	@note		Call such as Event::query() or Stream::waitEvent() will then examine or wait for completion of the work that was captured.
		 *	@note		Can be called multiple times on the same event and will overwrite the previously captured state.
		 *	@warning	Event and stream must be on the same device.
		 */
		void recordEvent(cudaEvent_t hEvent);


		/**
		 *	@brief		Make a compute stream wait on an event.
		 *	@param[in]	hEvent - Valid event handle to wait on.
		 *	@note		Make all future work submitted to this stream wait for all work captured in event.
		 *	@note		Event may be from a different device than this stream.
		 */
		void waitEvent(cudaEvent_t hEvent) const;


		/**
		 *	@brief		Launches an executable graph in a stream
		 *	@param[in]	hGraphExec - Executable graph to launch
		 *	@retval		Stream - Reference to this stream, convenient for further operations.
		 *	@note		Each launch is ordered behind both any previous work in this stream and any previous launches of graphExec.
		 *	@example	stream.launchGraph(hGraph)
		 *	@example	stream.launchGraph(hGraph).sync()
		 *	@example	stream.launchGraph(hGraph).launchGraph(hGraph)
		 */
		Stream & launchGraph(cudaGraphExec_t hGraphExec);


		/**
		 *	@brief		Enqueue a host function call in a stream.
		 *	@param[in]	func - The function to call once preceding stream operations are complete.
		 *	@param[in]	userData - User-specified data to be passed to the function.
		 *	@retval		Stream - Reference to this stream, convenient for further operations.
		 *	@note		Host function will be called from the thread named nvcuda64.dll.
		 *	@note		The function will be called after currently enqueued work and will block work added after it.
		 *	@warning	The host function must not perform any synchronization that may depend on outstanding CUDA work not mandated to run earlier.
		 *	@warning	The host function must not make any CUDA API calls.
		 */
		template<typename Type> Stream & launchHostFunc(HostFunc<Type> func, Type * userData)
		{
			return this->launchHostFunc(reinterpret_cast<HostFunc<void>>(func), userData);
		}
		Stream & launchHostFunc(HostFunc<void> func, void * userData);


		/**
		 *	@brief		Prepares to launch a CUDA kernel with specified parameters and dependencies.
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	gDim - Grid dimensions.
		 *	@param[in]	bDim - Block dimensions.
		 *	@param[in]	sharedMem - Number of bytes for shared memory.
		 *	@example	stream.launch(KernelAdd, gridDim, blockDim, sharedMem)(pResult, pA, pB, num);
		 *	@note		The returned lambda is a temporary object that should be used immediately to configure and launch the kernel.
		 *				It encapsulates all necessary information for the kernel launch, including the kernel function, its arguments.
		 *	@warning	Only available in *.cu files (implemented in stream.cuh).
		 */
		template<typename... Args> NS_NODISCARD auto launch(KernelFunc<Args...> func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem = 0);


		/**
		 *	@brief		Copies data between 3D objects.
		 *	@param[in]	dst - Destination memory address.
		 *	@param[in]	dstPitch - Pitch of destination memory.
		 *	@param[in]	dstHeight - Height of destination memory.
		 *	@param[in]	src - Source memory address.
		 *	@param[in]	srcPitch - Pitch of source memory.
		 *	@param[in]	srcHeight - Height of source memory.
		 *	@param[in]	width - Width of matrix transfer (columns in bytes).
		 *	@param[in]	height - Height of matrix transfer (rows).
		 *	@param[in]	depth - Depth of matrix transfer (layers).
		 *	@param[in]	extext - Extent of matrix transfer.
		 *	@warning	The memory areas may not overlap. \p width must not exceed either \p dstPitch or \p srcPitch.
		 */
		void memcpy3D(void * dst, size_t dstPitch, size_t dstHeight, const void * src, size_t srcPitch, size_t srcHeight, size_t width, size_t height, size_t depth);


		/**
		 *	@brief		Initialize or set device memory to a value.
		 *	@param[in]	pValues - Pointer to the device memory.
		 *	@param[in]	value - Value to set for.
		 *	@param[in]	count - Count of values to set.
		 *	@param[in]	blockSize - 
		 *	@retval		Stream - Reference to this stream, convenient for further operations.
		 *	@warning	Only available in *.cu files.
		 */
		template<typename Type> Stream & memset(Type * pValues, Type value, size_t count, int blockSize = 256);


		/**
		 *	@brief		Initialize or set device memory to zeros.
		 *	@param[in]	address - Pointer to device memory.
		 *	@param[in]	bytes - Size in bytes to set.
		 *	@retval		Stream - Reference to this stream, convenient for further operations.
		 */
		Stream & memsetZero(void * address, size_t bytes);
		
	private:

		/**
		 *	@brief		Launches a device function
		 *	@param[in]	func - Device function symbol.
		 *	@param[in]	gDim - Grid dimensions.
		 *	@param[in]	bDim - Block dimensions.
		 *	@param[in]	sharedMemBytes - Number of bytes for shared memory.
		 *	@param[in]	args - Pointers to kernel arguments.
		 *	@warning	Only available in *.cu files.
		 */
		void launchKernel(const void * func, const dim3 & gridDim, const dim3 & blockDim, size_t sharedMem, void ** args);


		void acquireDeviceContext() const;

	private:

		Device * const				m_device;

		cudaStream_t  				m_hStream;

		int							m_priority;
	};
}