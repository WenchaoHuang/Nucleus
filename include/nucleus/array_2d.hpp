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

#include "fwd.hpp"
#include "dev_ptr.hpp"
#include "allocator.hpp"

namespace NS_NAMESPACE
{
	/*********************************************************************
	***************************    Array2D    ****************************
	*********************************************************************/

	/**
	 *	@brief		Template for 2D array, which accessible to the device.
	 */
	template<typename Type> class Array2D
	{
		NS_NONCOPYABLE(Array2D)

	public:

		//!	@brief		Construct an empty array.
		Array2D() : m_allocator(nullptr), m_data(nullptr), m_width(0), m_height(0) {}

		//!	@brief		Allocates 2d array with \p width * \p height elements.
		explicit Array2D(std::shared_ptr<Allocator> alloctor, size_t width, size_t height) : Array2D() { this->reshape(alloctor, width, height); }

		//!	@brief		Move constructor.
		Array2D(Array2D && rhs) : m_allocator(std::move(rhs.m_allocator)), m_data(rhs.m_data), m_width(rhs.m_width), m_height(rhs.m_height)
		{
			rhs.m_data = nullptr;	rhs.m_width = rhs.m_height = 0;
		}

		//!	@brief		Destroy array.
		~Array2D() { this->clear(); }

	public:

		//!	@brief		Returns width of 2D array.
		uint32_t width() const { return m_width; }

		//!	@brief		Returns height of 2D array.
		uint32_t height() const { return m_height; }

		//!	@brief		Tests if the array is empty.
		bool empty() const { return m_data == nullptr; }

		//!	@brief		Returns count of elements allocated on device.
		uint32_t size() const { return m_height * m_width; }

		//!	@brief		Returns pitch of 2D array in bytes.
		size_t pitch() const { return m_width * sizeof(Type); }

		//!	@brief		Returns bytes of memory allocated. 
		size_t bytes() const { return m_height * m_width * sizeof(Type); }

		//!	@brief		Returns the allocator associated with.
		std::shared_ptr<Allocator> getAllocator() const { return m_allocator; }

		//!	@brief		Returns device pointer to the specified row of the 3D array (const version).
		DevPtr<const Type> operator[](size_t row) const { NS_ASSERT(row < m_height);	return DevPtr<const Type>(m_data + row * m_width); }

		//!	@brief		Returns device pointer to the specified row of the 3D array.
		DevPtr<Type> operator[](size_t row) { NS_ASSERT(row < m_height);	return DevPtr<Type>(m_data + row * m_width); }

		//!	@brief		Returns device pointer to the underlying array (const version).
		DevPtr2<const Type> ptr() const { return DevPtr2<const Type>(m_data, m_width, m_height); }

		//!	@brief		Returns device pointer to the underlying array.
		DevPtr2<Type> ptr() { return DevPtr2<Type>(m_data, m_width, m_height); }

		//!	@brief		Returns raw pointer to the data (const version).
		const Type * rawPtr() const { return m_data; }

		//!	@brief		Returns raw pointer to the data.
		Type * rawPtr() { return m_data; }

		/**
		 *	@brief		Changes the number of elements stored.
		 */
		void reshape(std::shared_ptr<Allocator> alloctor, size_t width, size_t height)
		{
			NS_ASSERT((alloctor != nullptr) && (width < UINT32_MAX) && (height < UINT32_MAX));

			if ((m_allocator != alloctor) || (width * height != m_width * m_height))
			{
				this->clear();

				if (width * height > 0)
				{
					m_data = static_cast<Type*>(alloctor->allocateMemory(sizeof(Type) * width * height));

					m_height = static_cast<uint32_t>(height);

					m_width = static_cast<uint32_t>(width);

					m_allocator = alloctor;
				}
			}
			else if ((m_width != width) || (m_height != height))
			{
				m_height = static_cast<uint32_t>(height);

				m_width = static_cast<uint32_t>(width);
			}
		}


		/**
		 *	@brief		Swaps the contents.
		 */
		void swap(Array2D & rhs) noexcept
		{
			std::swap(m_allocator, rhs.m_allocator);

			std::swap(m_height, rhs.m_height);

			std::swap(m_width, rhs.m_width);

			std::swap(m_data, rhs.m_data);
		}


		/**
		 *	@brief		Clears the contents.
		 */
		void clear()
		{
			if (m_data != nullptr)
			{
				m_allocator->deallocateMemory(m_data);

				m_height = m_width = 0;

				m_allocator = nullptr;

				m_data = nullptr;
			}
		}

	private:

		Type *							m_data;
		uint32_t						m_width;
		uint32_t						m_height;
		std::shared_ptr<Allocator>		m_allocator;
	};
}