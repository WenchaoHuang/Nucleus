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
#include "vector_types.h"

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Format    ****************************
	*********************************************************************/

	/**
	 *	@brief	Available texel formats for CUDA texture objects.
	 *	@note	This enum class defines the supported data types and component counts
	 *			for texels in CUDA texture objects. Each format specifies a base type
	 *			(e.g., char, int, float) and the number of components (1, 2, or 4).
	 */
	enum class Format
	{
		Int, Int2, Int4,			// Signed 32-bit integer (1, 2, or 4 components)
		Uint, Uint2, Uint4,			// Unsigned 32-bit integer (1, 2, or 4 components)
		Char, Char2, Char4,			// Signed 8-bit integer (1, 2, or 4 components)
		Uchar, Uchar2, Uchar4,		// Unsigned 8-bit integer (1, 2, or 4 components)
		Short, Short2, Short4,		// Signed 16-bit integer (1, 2, or 4 components)
		Ushort, Ushort2, Ushort4,	// Unsigned 16-bit integer (1, 2, or 4 components)
		Float, Float2, Float4,		// 32-bit floating-point (1, 2, or 4 components)
	};

	/*********************************************************************
	*****************    TypeMapping / FormatMapping    ******************
	*********************************************************************/

	namespace details
	{
		template<Format format> struct TypeMapping;
		template<typename Type> struct FormatMapping;

		//	Maps Format enum values to CUDA internal types.
		template<> struct TypeMapping<Format::Int>  { using type = int; };
		template<> struct TypeMapping<Format::Int2> { using type = int2; };
		template<> struct TypeMapping<Format::Int4> { using type = int4; };

		template<> struct TypeMapping<Format::Uint>  { using type = uint; };
		template<> struct TypeMapping<Format::Uint2> { using type = uint2; };
		template<> struct TypeMapping<Format::Uint4> { using type = uint4; };

		template<> struct TypeMapping<Format::Char>  { using type = char; };
		template<> struct TypeMapping<Format::Char2> { using type = char2; };
		template<> struct TypeMapping<Format::Char4> { using type = char4; };

		template<> struct TypeMapping<Format::Uchar>  { using type = uchar; };
		template<> struct TypeMapping<Format::Uchar2> { using type = uchar2; };
		template<> struct TypeMapping<Format::Uchar4> { using type = uchar4; };

		template<> struct TypeMapping<Format::Short>  { using type = short; };
		template<> struct TypeMapping<Format::Short2> { using type = short2; };
		template<> struct TypeMapping<Format::Short4> { using type = short4; };

		template<> struct TypeMapping<Format::Ushort>  { using type = ushort; };
		template<> struct TypeMapping<Format::Ushort2> { using type = ushort2; };
		template<> struct TypeMapping<Format::Ushort4> { using type = ushort4; };

		template<> struct TypeMapping<Format::Float>  { using type = float; };
		template<> struct TypeMapping<Format::Float2> { using type = float2; };
		template<> struct TypeMapping<Format::Float4> { using type = float4; };

		//	Maps C++ types to corresponding Format enum values.
		template<> struct FormatMapping<int>  { static constexpr Format value = Format::Int; };
		template<> struct FormatMapping<int2> { static constexpr Format value = Format::Int2; };
		template<> struct FormatMapping<int4> { static constexpr Format value = Format::Int4; };

		template<> struct FormatMapping<uint>  { static constexpr Format value = Format::Uint; };
		template<> struct FormatMapping<uint2> { static constexpr Format value = Format::Uint2; };
		template<> struct FormatMapping<uint4> { static constexpr Format value = Format::Uint4; };

		template<> struct FormatMapping<char>  { static constexpr Format value = Format::Char; };
		template<> struct FormatMapping<char2> { static constexpr Format value = Format::Char2; };
		template<> struct FormatMapping<char4> { static constexpr Format value = Format::Char4; };

		template<> struct FormatMapping<uchar>  { static constexpr Format value = Format::Uchar; };
		template<> struct FormatMapping<uchar2> { static constexpr Format value = Format::Uchar2; };
		template<> struct FormatMapping<uchar4> { static constexpr Format value = Format::Uchar4; };

		template<> struct FormatMapping<short>  { static constexpr Format value = Format::Short; };
		template<> struct FormatMapping<short2> { static constexpr Format value = Format::Short2; };
		template<> struct FormatMapping<short4> { static constexpr Format value = Format::Short4; };

		template<> struct FormatMapping<ushort>  { static constexpr Format value = Format::Ushort; };
		template<> struct FormatMapping<ushort2> { static constexpr Format value = Format::Ushort2; };
		template<> struct FormatMapping<ushort4> { static constexpr Format value = Format::Ushort4; };

		template<> struct FormatMapping<float>  { static constexpr Format value = Format::Float; };
		template<> struct FormatMapping<float2> { static constexpr Format value = Format::Float2; };
		template<> struct FormatMapping<float4> { static constexpr Format value = Format::Float4; };
	}

	/*********************************************************************
	************************    FormatMapping    *************************
	*********************************************************************/

	/**
	 *	@brief	Wrapper for format mapping, defaults to FormatMapping.
	 *	@note	Allows customization of format mappings while providing default mappings
	 *			for standard types. Used to associate types with Format enum values.
	 */
	template<typename Type> struct FormatMapping
	{
		static constexpr Format value = details::FormatMapping<Type>::value;
	};

	//	Defines the internal CUDA-compatible value type for a given C++ type.
	template<typename Type> using InternalValueType = typename details::TypeMapping<FormatMapping<std::remove_const_t<Type>>::value>::type;

	/*********************************************************************
	*********************    IsValidFormatMapping    *********************
	*********************************************************************/

	/**
	 *	@brief	Validates type mappings at compile-time for CUDA texture configurations.
	 *	@note	Uses static assertions to ensure that the size and alignment of Type match
	 *			those of the type mapped from its corresponding Format enum value. This
	 *			ensures type safety and consistency in CUDA texture object configurations.
	 */
	template<typename Type> struct CheckFormatMapping
	{
		using underlying_type = typename details::TypeMapping<FormatMapping<Type>::value>::type;

		static_assert(sizeof(Type) == sizeof(underlying_type), "Size mismatch in format mapping!");
		static_assert(alignof(Type) == alignof(underlying_type), "Alignment mismatch in format mapping!");
	};
}