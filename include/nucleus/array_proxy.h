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
#include <array>
#include <vector>

namespace NS_NAMESPACE
{
	/*****************************************************************************
	******************************    ArrayProxy    ******************************
	*****************************************************************************/

	/**
	 *	@brief		A lightweight non-owning view over a contiguous array of elements.
	 *
	 *	This class provides a convenient way to pass around contiguous blocks of memory (arrays, vectors, std::array, etc.)
	 *	without copying the data. It behaves similarly to `std::span` (introduced in C++20), but with a simplified interface
	 *	and limited to const access.
	 *
	 *	@tparam Type - The element type, which may be const or non-const. Internally, the value type is defined as
	 *	              `std::remove_const_t<Type>`, allowing consistent construction from various containers.
	 *
	 *	Key Features:
	 *	- Can be constructed from raw pointers, fixed-size arrays, `std::initializer_list`, `std::array`, or `std::vector`.
	 *	- Provides const access to the underlying data.
	 *	- Supports bounds-checked element access via `operator[]`, as well as `front()`, `back()`, `begin()`, and `end()`.
	 *	- Used mainly as a parameter or temporary wrapper to avoid ownership and unnecessary memory copies.
	 *
	 *	Typical usage:
	 *	@code
	 *		void process(ArrayProxy<const float> data);
	 *
	 *		std::vector<float> values = {1.0f, 2.0f, 3.0f};
	 *		process(values);
	 *
	 *		float arr[2] = {4.0f, 5.0f};
	 *		process(arr);
	 *	@endcode
	 */
	template<typename Type> class ArrayProxy
	{

	public:

		//!	@brief		Defines the value type.
		using value_type = std::remove_const_t<Type>;

		//!	@brief		Default constructor.
		ArrayProxy() noexcept : m_data(nullptr), m_count(0) {}

		//!	@brief		Only used in this case: `ArrayProxy<T> = nullptr`.
		ArrayProxy(std::nullptr_t) noexcept : m_data(nullptr), m_count(0) {}

		//!	@brief		Construct with an instance.
		ArrayProxy(const value_type & inst) noexcept : m_data(&inst), m_count(1) {}

		//!	@brief		Construct with a list of data.
		explicit ArrayProxy(const value_type * data, uint32_t count) noexcept : m_data(data), m_count(count) {}

		//!	@brief		Construct with `std::initializer_list<value_type>`.
		ArrayProxy(std::initializer_list<value_type> list) : m_data(list.begin()), m_count(static_cast<uint32_t>(list.size())) {}

		//!	@brief		Construct with fixed-size array.
		template<size_t N> ArrayProxy(const value_type(&array)[N]) noexcept : m_data(array), m_count(static_cast<uint32_t>(N)) {}

		//!	@brief		Construct with `std::array<value_type, N>`.
		template<size_t N> ArrayProxy(const std::array<value_type, N> & array) : m_data(array.data()), m_count(static_cast<uint32_t>(array.size())) {}

		//!	@brief		Construct with `std::vector<value_type, Alloc>`.
		template<class Alloc> ArrayProxy(const std::vector<value_type, Alloc> & vector) : m_data(vector.data()), m_count(static_cast<uint32_t>(vector.size())) {}

	public:

		//!	@brief		Test if the array is empty.
		bool empty() const noexcept { return (m_data == nullptr) || (m_count == 0); }

		//!	@brief		Return reference to the specified element.
		const Type & operator[](size_t pos) const { assert(pos < m_count); return m_data[pos]; }

		//!	@brief		Return last element of this array.
		const Type & back() const { assert(m_count > 0); return *(m_data + m_count - 1); }

		//!	@brief		Return first element of this array.
		const Type & front() const { assert(m_count > 0); return *m_data; }

		//!	@brief		Get end of the list.
		const Type * end() const { return m_data + m_count; }

		//!	@brief		Get beginning of the list.
		const Type * begin() const { return m_data; }

		//!	@brief		Return address to the first element.
		const Type * data() const { return m_data; }

		//!	@brief		Return count of elements.
		uint32_t size() const { return m_count; }

	private:

		const Type *	m_data;

		uint32_t		m_count;
	};
}