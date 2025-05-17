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
	***************************    Array3D    ****************************
	*********************************************************************/

	/**
	 *	@brief		Template for 3D array, which accessible to the device.
	 */
	template<typename Type> class Array3D
	{
		NS_NONCOPYABLE(Array3D)

	public:

		//!	@brief		Construct an empty array.
		Array3D() : m_allocator(nullptr), m_data(nullptr), m_width(0), m_height(0), m_depth(0) {}

		//!	@brief		Allocates 3d array with \p width * \p height * \p depth elements.
		explicit Array3D(std::shared_ptr<Allocator> pAlloc, size_t width, size_t height, size_t depth) : Array3D() { this->reshape(pAlloc, width, height, depth); }

		//!	@brief		Move constructor.
		Array3D(Array3D && rhs) : m_allocator(std::move(rhs.m_allocator)), m_data(rhs.m_data), m_width(rhs.m_width), m_height(rhs.m_height), m_depth(rhs.m_depth)
		{
			rhs.m_data = nullptr;	rhs.m_width = rhs.m_height = rhs.m_depth = 0;
		}

		//!	@brief		Destroy array.
		~Array3D() { this->clear(); }

	public:

		//!	@brief		Returns width of 3D array.
		uint32_t width() const { return m_width; }

		//!	@brief		Returns depth of 3D array.
		uint32_t depth() const { return m_depth; }

		//!	@brief		Returns height of 3D array.
		uint32_t height() const { return m_height; }

		//!	@brief		Tests if the array is empty.
		bool empty() const { return m_data == nullptr; }

		//!	@brief		Returns pitch of 3D array in bytes.
		size_t pitch() const { return m_width * sizeof(Type); }

		//!	@brief		Returns count of elements allocated on device.
		uint32_t size() const { return m_width * m_height * m_depth; }

		//!	@brief		Returns the allocator associated with.
		std::shared_ptr<Allocator> getAllocator() const { return m_allocator; }

		//!	@brief		Returns bytes of memory allocated. 
		size_t bytes() const { return m_width * m_height * m_depth * sizeof(Type); }

		//!	@brief		Returns device pointer to the specified layer of the 3D array (const version).
		DevPtr2<const Type> operator[](size_t layer) const { NS_ASSERT(layer < m_depth);	return DevPtr2<const Type>(m_data + layer * (m_width * m_height), m_width, m_height); }

		//!	@brief		Returns device pointer to the specified layer of the 3D array.
		DevPtr2<Type> operator[](size_t layer) { NS_ASSERT(layer < m_depth);	return DevPtr2<Type>(m_data + layer * (m_width * m_height), m_width, m_height); }

		//!	@brief		Returns device pointer to the underlying 3D array (const version).
		DevPtr3<const Type> ptr() const { return DevPtr3<const Type>(m_data, m_width, m_height, m_depth); }

		//!	@brief		Returns device pointer to the underlying 3D array.
		DevPtr3<Type> ptr() { return DevPtr3<Type>(m_data, m_width, m_height, m_depth); }

		//!	@brief		Returns raw pointer to the data (const version).
		const Type * rawPtr() const { return m_data; }

		//!	@brief		Returns raw pointer to the data.
		Type * rawPtr() { return m_data; }

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