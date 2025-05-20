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

#include "allocator.h"

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Buffer    ****************************
	*********************************************************************/

	/**
	 *	@brief		RAII wrapper for memory management.
	 *	@note		This class provides a safe and exception-safe way to manage dynamically
	 *				allocated memory buffers. It handles allocation and deallocation
	 *				automatically through the provided Allocator interface.
	 */
	class Buffer
	{
		NS_NONCOPYABLE(Buffer)

	public:

		/**
		 *	@brief		Constructs an empty buffer.
		 *	@note		Creates a Buffer object with null data pointer and zero capacity.
		 *				No memory allocation is performed.
		 */
		Buffer() : m_allocator(nullptr), m_data(nullptr), m_capacity(0) {}


		/**
		 *	@brief		Constructs a buffer with specified capacity.
		 *	@param[in]	allocator - Shared pointer to the memory allocator.
		 *	@param[in]	capacity - Requested buffer capacity in bytes.
		 *	@throws		cudaError_t - In case of failure.
		 *	@note		The actual allocation is delegated to the provided Allocator
		 */
		explicit Buffer(std::shared_ptr<Allocator> allocator, size_t capacity) : m_allocator(allocator), m_capacity(capacity), m_data(nullptr)
		{
			m_data = allocator->allocateMemory(capacity);
		}


		/**
		 *	@brief		Destructor.
		 *	@details	Automatically deallocates the buffer using the stored allocator.
		 *				Guaranteed not to throw exceptions (noexcept).
		 */
		~Buffer() noexcept
		{
			if ((m_allocator != nullptr) && (m_data != nullptr))
			{
				m_allocator->deallocateMemory(m_data);

				m_allocator = nullptr;

				m_data = nullptr;

				m_capacity = 0;
			}
		}

	public:

		/**
		 *	@brief		Gets the associated memory allocator.
		 *	@return		Shared pointer to the Allocator instance.
		 */
		std::shared_ptr<Allocator> getAllocator() const { return m_allocator; }


		/**
		 *	@brief		Checks if buffer is empty.
		 *	@return		Returns true if buffer is empty (nullptr), false otherwise.
		 */
		bool empty() const { return (m_data == nullptr); }


		/**
		 *	@brief		Gets the buffer capacity.
		 *	@return		Returns the total number of bytes allocated for this buffer.
		 */
		size_t capacity() const { return m_capacity; }


		/**
		 *	@brief		Retrun logical address of the memory.
		 */
		std::uintptr_t address() const { return reinterpret_cast<std::uintptr_t>(m_data); }

		/**
		 *	@brief		Gets constant pointer to the underlying data.
		 *	@return		Const pointer to the buffer data
		 *	@note		The returned pointer remains valid until the Buffer object is destroyed.
		 */
		const void * data() const { return m_data; }


		/**
		 *	@brief		Gets mutable pointer to the underlying data.
		 *	@return		Pointer to the buffer data
		 *	@note		The returned pointer remains valid until the Buffer object is destroyed.
		 */
		void * data() { return m_data; }
		
	private:

		void *							m_data;
		size_t							m_capacity;
		std::shared_ptr<Allocator>		m_allocator;
	};
}