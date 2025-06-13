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
	/*********************************************************************
	**************************    Allocator    ***************************
	*********************************************************************/

	/**
	 *	@brief		An abstract interface to an unbounded set of classes
	 *				encapsulating memory resources accessible to the device.
	 */
	class NS_NOVTABLE Allocator
	{
		NS_NONCOPYABLE(Allocator)

	public:

		//!	@brief		Construct Allocator.
		Allocator() noexcept = default;

		//!	@brief		Destruct Allocator.
		virtual ~Allocator() noexcept {}

	public:

		/**
		 *	@brief		Allocate storage with a size of at least \p bytes
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to newly allocated memory.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_NODISCARD void * allocateMemory(size_t bytes) { return this->doAllocateMemory(bytes); }


		/**
		 *	@brief		Deallocates memory previously allocated by Allocator::Allocate().
		 *	@param[in]	ptr - A pointer to the memory to be deallocated. This must be a pointer obtained from a prior
		 *				call to Allocator::Allocate(), and the memory it points to must not yet have been deallocated.
		 */
		void deallocateMemory(void * ptr) { this->doDeallocateMemory(ptr); }

	private:

		//!	@brief		Allocates memory.
		virtual void * doAllocateMemory(size_t bytes) = 0;

		//!	@brief		Deallocates memory.
		virtual void doDeallocateMemory(void * ptr) = 0;
	};

	/*********************************************************************
	************************    HostAllocator    *************************
	*********************************************************************/

	/**
	 *	@brief		Page-locked or pinned memory allocator.
	 *	@note		Since the memory can be accessed directly by the device,
	 *				it can be read or written with much higher bandwidth than pageable memory.
	 *	@note		Page-locked memory is best used sparingly to allocate staging areas for data exchange between host and device.
	 */
	class HostAllocator : public Allocator
	{

	protected:

		/**
		 *	@brief		Allocate page-locked memory on the host.
		 *	@note		Allocate host memory that is page-locked and accessible to the device.
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to newly allocated memory.
		 *	@details	Memories allocated by CUDA API are always aligned to 512B.
		 *	@details	If \p bytes is 0, nullptr will be returned.
		 *	@throw		cudaError_t - In case of failure.
		 */
		virtual void * doAllocateMemory(size_t bytes) override;


		/**
		 *	@brief		Deallocate page-locked memory.
		 *	@param[in]	ptr - Memory address returned by HostAllocator::do_allocate.
		 *	@details	If \p ptr is nullptr, no operation is performed.
		 */
		virtual void doDeallocateMemory(void * ptr) override;
	};

	/*********************************************************************
	***********************    DeviceAllocator    ************************
	*********************************************************************/

	/**
	 *	@brief		Device memory allocator.
	 */
	class DeviceAllocator : public Allocator
	{

	public:

		//!	@brief		Constructs device memory allocator.
		explicit DeviceAllocator(Device * pDevice);

		//!	@brief		Return pointer to the device associated with.
		Device * getDevice() const { return m_device; }

	public:

		/**
		 *	@brief		Allocate texture memory on the device.
		 *	@param[in]	eFormat - Texel format of the buffer.
		 *	@param[in]	width - width of the buffer.
		 *	@param[in]	height - height of the buffer.
		 *	@param[in]	depth - depth of the buffer.
		 *	@param[in]	flags - Flags for buffer creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		virtual cudaArray_t allocateTextureMemory(Format eFormat, size_t width, size_t height, size_t depth, int flags = 0);


		/**
		 *	@brief		Free texture memory on the device.
		 *	@param[in]	hArray - Returned by a previous call to DeviceAllocator::allocateTextureMemory().
		 *	@details	If \p hArray is nullptr, no operation is performed.
		 */
		virtual void deallocateTextureMemory(cudaArray_t hArray);

	public:

		/**
		 *	@brief		Allocate mipmapped texture memory on the device.
		 *	@param[in]	eFormat - Texel format of the buffer.
		 *	@param[in]	width - width of the buffer.
		 *	@param[in]	height - height of the buffer.
		 *	@param[in]	depth - depth of the buffer.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated.
		 *	@param[in]	flags - Flags for buffer creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		virtual cudaMipmappedArray_t allocateMipmapTextureMemory(Format eFormat, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags = 0);


		/**
		 *	@brief		Free mipmapped texture memory on the device.
		 *	@param[in]	hMipmapedArray - Returned by a previous call to DeviceAllocator::allocateMipmapTextureMemory().
		 *	@details	If \p hMipmapedArray is nullptr, no operation is performed.
		 */
		virtual void deallocateMipmapTextureMemory(cudaMipmappedArray_t hMipmapedArray);

	protected:

		/**
		 *	@brief		Allocate linear memory on device.
		 *	@param[in]	bytes - Requested allocation size in bytes.
		 *	@return		Pointer to the newly allocated memory.
		 *	@details	Memories allocated by CUDA API are always aligned to 512B.
		 *	@details	If \p bytes is 0, nullptr will be returned.
		 *	@throw		cudaError_t - In case of failure.
		 */
		virtual void * doAllocateMemory(size_t bytes) override;


		/**
		 *	@brief		Free memory on device.
		 *	@param[in]	ptr - Memory address returned by DeviceAllocator::Allocate().
		 *	@details	If \p ptr is nullptr, no operation is performed.
		 */
		virtual void doDeallocateMemory(void * ptr) override;

	private:

		Device * const		m_device;
	};
}