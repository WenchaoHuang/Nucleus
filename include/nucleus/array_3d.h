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
#include "dev_ptr.h"
#include "allocator.h"

namespace NS_NAMESPACE
{
	/*********************************************************************
	***************************    Array3D    ****************************
	*********************************************************************/

	/**
	 *	@brief		Template for 3D array, which accessible to the device.
	 */
	template<typename Type> class Array3D : public DevPtr3<Type>
	{
		NS_NONCOPYABLE(Array3D)

	public:

		//!	@brief		Construct an empty array.
		Array3D() noexcept : m_allocator(nullptr), m_data(nullptr), m_width(0), m_height(0), m_depth(0) {}

		//!	@brief		Allocates 3d array with \p width * \p height * \p depth elements.
		explicit Array3D(std::shared_ptr<Allocator> pAlloc, size_t width, size_t height, size_t depth) : Array3D() { this->reshape(pAlloc, width, height, depth); }

		//!	@brief		Move constructor.
		Array3D(Array3D && rhs) noexcept : m_allocator(std::exchange(rhs.m_allocator, nullptr)), m_data(std::exchange(rhs.m_data, nullptr)), m_width(std::exchange(rhs.m_width, 0)), m_height(std::exchange(rhs.m_height, 0)), m_depth(std::exchange(rhs.m_depth, 0))
		{}

		//!	@brief		Move assignment.
		void operator=(Array3D && rhs) noexcept
		{
			m_allocator = std::exchange(rhs.m_allocator, nullptr);

			m_data = std::exchange(rhs.m_data, nullptr);

			m_height = std::exchange(rhs.m_height, 0);

			m_width = std::exchange(rhs.m_width, 0);

			m_depth = std::exchange(rhs.m_depth, 0);
		}

		//!	@brief		Destroy array.
		~Array3D() { this->clear(); }

	public:

		/**
		 *	@brief		Returns the allocator associated with.
		 */
		const std::shared_ptr<Allocator> & getAllocator() const { return m_allocator; }


		/**
		 *	@brief		Changes the number of elements stored.
		 */
		void reshape(std::shared_ptr<Allocator> allocator, size_t width, size_t height, size_t depth)
		{
			NS_ASSERT((allocator != nullptr) && (width < UINT32_MAX) && (height < UINT32_MAX) && (depth < UINT32_MAX));

			if ((m_allocator != allocator) || (width * height * depth != this->size()))
			{
				this->clear();

				if (width * height * depth > 0)
				{
					m_data = static_cast<Type*>(allocator->allocateMemory(sizeof(Type) * width * height * depth));

					m_height = static_cast<uint32_t>(height);

					m_width = static_cast<uint32_t>(width);

					m_depth = static_cast<uint32_t>(depth);

					m_allocator = allocator;
				}
			}
			else if ((m_width != width) || (m_height != height) || (m_depth != depth))
			{
				m_height = static_cast<uint32_t>(height);

				m_width = static_cast<uint32_t>(width);

				m_depth = static_cast<uint32_t>(depth);
			}
		}


		/**
		 *	@brief		Swaps the contents.
		 */
		void swap(Array3D & rhs) noexcept
		{
			std::swap(m_allocator, rhs.m_allocator);

			std::swap(m_height, rhs.m_height);

			std::swap(m_width, rhs.m_width);

			std::swap(m_depth, rhs.m_depth);

			std::swap(m_data, rhs.m_data);
		}


		/**
		 *	@brief		Clears the contents.
		 */
		void clear()
		{
			if (m_allocator != nullptr)
			{
				m_allocator->deallocateMemory(m_data);

				m_width = m_height = m_depth = 0;

				m_allocator = nullptr;

				m_data = nullptr;
			}
		}

	private:

		Type *							m_data;
		uint32_t						m_width;
		uint32_t						m_height;
		uint32_t						m_depth;
		std::shared_ptr<Allocator>		m_allocator;
	};
}