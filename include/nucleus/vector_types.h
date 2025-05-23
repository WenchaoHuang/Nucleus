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

#include "macros.h"

namespace NS_NAMESPACE
{
	/*********************************************************************
	************************    Vector Types     *************************
	*********************************************************************/

	/**
	 *	@brief	Type aliases for commonly used vector types, conditionally defined.
	 * 
	 *	In non-CUDA environments (__CUDACC__ not defined), custom Vec1, Vec2, Vec3,
	 *	and Vec4 templates are used to define vector types, ensuring no dependency on
	 *	CUDA headers. This provides a consistent interface for scalar and vector types
	 *	across platforms, enhancing modularity and portability.
	 * 
	 *	This approach avoids exposing CUDA headers in the interface, allowing the
	 *	header to be included in non-CUDA projects.
	 */
#ifndef __CUDACC__
	template<typename Type> struct Vec1 { Type x; };
	template<typename Type> struct Vec3 { Type x, y, z; };
	template<typename Type> struct NS_ALIGN(NS_MIN(alignof(Type) * 2, 16)) Vec2 { Type x, y; };
	template<typename Type> struct NS_ALIGN(NS_MIN(alignof(Type) * 4, 16)) Vec4 { Type x, y, z, w; };

	using int1 = Vec1<int>;
	using int2 = Vec2<int>;
	using int3 = Vec3<int>;
	using int4 = Vec4<int>;

	using char1 = Vec1<char>;
	using char2 = Vec2<char>;
	using char3 = Vec3<char>;
	using char4 = Vec4<char>;

	using short1 = Vec1<short>;
	using short2 = Vec2<short>;
	using short3 = Vec3<short>;
	using short4 = Vec4<short>;

	using float1 = Vec1<float>;
	using float2 = Vec2<float>;
	using float3 = Vec3<float>;
	using float4 = Vec4<float>;

	using double1 = Vec1<double>;
	using double2 = Vec2<double>;
	using double3 = Vec3<double>;
	using double4 = Vec4<double>;

	using uint = unsigned int;
	using uint1 = Vec1<unsigned int>;
	using uint2 = Vec2<unsigned int>;
	using uint3 = Vec3<unsigned int>;
	using uint4 = Vec4<unsigned int>;

	using longlong = long long;
	using longlong1 = Vec1<long long>;
	using longlong2 = Vec2<long long>;
	using longlong3 = Vec3<long long>;
	using longlong4 = Vec4<long long>;

	using uchar = unsigned char;
	using uchar1 = Vec1<unsigned char>;
	using uchar2 = Vec2<unsigned char>;
	using uchar3 = Vec3<unsigned char>;
	using uchar4 = Vec4<unsigned char>;

	using ushort = unsigned short;
	using ushort1 = Vec1<unsigned short>;
	using ushort2 = Vec2<unsigned short>;
	using ushort3 = Vec3<unsigned short>;
	using ushort4 = Vec4<unsigned short>;

	using ulonglong = unsigned long long;
	using ulonglong1 = Vec1<unsigned long long>;
	using ulonglong2 = Vec2<unsigned long long>;
	using ulonglong3 = Vec3<unsigned long long>;
	using ulonglong4 = Vec4<unsigned long long>;
#else
	using int1 = ::int1;
	using int2 = ::int2;
	using int3 = ::int3;
	using int4 = ::int4;

	using uint = unsigned int;
	using uint1 = ::uint1;
	using uint2 = ::uint2;
	using uint3 = ::uint3;
	using uint4 = ::uint4;
	
	using char1 = ::char1;
	using char2 = ::char2;
	using char3 = ::char3;
	using char4 = ::char4;

	using uchar = unsigned char;
	using uchar1 = ::uchar1;
	using uchar2 = ::uchar2;
	using uchar3 = ::uchar3;
	using uchar4 = ::uchar4;

	using short1 = ::short1;
	using short2 = ::short2;
	using short3 = ::short3;
	using short4 = ::short4;

	using float1 = ::float1;
	using float2 = ::float2;
	using float3 = ::float3;
	using float4 = ::float4;

	using double1 = ::double1;
	using double2 = ::double2;
	using double3 = ::double3;
	using double4 = ::double4;

	using ushort = unsigned short;
	using ushort1 = ::ushort1;
	using ushort2 = ::ushort2;
	using ushort3 = ::ushort3;
	using ushort4 = ::ushort4;

	using longlong = long long;
	using longlong1 = ::longlong1;
	using longlong2 = ::longlong2;
	using longlong3 = ::longlong3;
	using longlong4 = ::longlong4;

	using ulonglong = unsigned long long;
	using ulonglong1 = ::ulonglong1;
	using ulonglong2 = ::ulonglong2;
	using ulonglong3 = ::ulonglong3;
	using ulonglong4 = ::ulonglong4;
#endif
}