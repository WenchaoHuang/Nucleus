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
#include "allocator.h"
#include "device_pointer.h"

namespace NS_NAMESPACE
{
	/*********************************************************************
	***************************    Array1D    ****************************
	*********************************************************************/

	/**
	 *	@brief		Template for 1D array, which accessible to the device.
	 */
	template<typename Type> class Array1D : public dev::Ptr<Type>
	{
		NS_NONCOPYABLE(Array1D)

	public:

		//!	@brief		Construct an empty array.
		Array1D() noexcept : m_allocator(nullptr), m_data(nullptr), m_width(0) {}

		//!	@brief		Allocates array with \p width elements.
		explicit Array1D(std::shared_ptr<Allocator> alloctor, size_t width) : Array1D()
		{
			this->reshape(alloctor, width);
		}

		//!	@brief		Move constructor.
		Array1D(Array1D && rhs) noexcept : m_allocator(std::exchange(rhs.m_allocator, nullptr)), m_data(std::exchange(rhs.m_data, nullptr)), m_width(std::exchange(rhs.m_width, 0))
		{}

		//!	@brief		Move assignment.
		void operator=(Array1D && rhs) noexcept
		{
			m_allocator = std::exchange(rhs.m_allocator, nullptr);

			m_data = std::exchange(rhs.m_data, nullptr);

			m_width = std::exchange(rhs.m_width, 0);
		}

		//!	@brief		Destroy array.
		~Array1D() { this->clear(); }

	public:

		/**
		 *	@brief		Returns the allocator associated with.
		 */
		const std::shared_ptr<Allocator> & getAllocator() const { return m_allocator; }


		/**
		 *	@brief		Changes the number of elements stored.
		 */
		void reshape(std::shared_ptr<Allocator> alloctor, size_t width)
		{
			NS_ASSERT(alloctor != nullptr);

			if ((m_allocator != alloctor) || (m_width != width))
			{
				this->clear();

				if (width > 0)
				{
					m_data = static_cast<Type*>(alloctor->allocateMemory(sizeof(Type) * width));

					m_allocator = alloctor;

					m_width = width;
				}
			}
		}


		/**
		 *	@brief		Swaps the contents.
		 */
		void swap(Array1D & rhs) noexcept
		{
			std::swap(m_allocator, rhs.m_allocator);

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

				m_allocator = nullptr;

				m_data = nullptr;

				m_width = 0;
			}
		}

	private:

		Type *							m_data;
		size_t							m_width;
		std::shared_ptr<Allocator>		m_allocator;
	};
}