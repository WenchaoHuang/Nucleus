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
	/*****************************************************************************
	****************************    Vector Types     *****************************
	*****************************************************************************/

	/**
	 *	@brief	Type aliases for commonly used vector types, conditionally defined.
	 * 
	 *	In non-CUDA environments (__CUDACC__ not defined), custom Vec2, Vec3,
	 *	and Vec4 templates are used to define vector types, ensuring no dependency on
	 *	CUDA headers. This provides a consistent interface for scalar and vector types
	 *	across platforms, enhancing modularity and portability.
	 * 
	 *	This approach avoids exposing CUDA headers in the interface, allowing the
	 *	header to be included in non-CUDA projects.
	 */
	template<typename Type, int Align> struct Vec2 { Type x, y; };
	template<typename Type, int Align> struct Vec3 { Type x, y, z; };
	template<typename Type, int Align> struct Vec4 { Type x, y, z, w; };
#ifndef __CUDACC__
	using int2 = Vec2<int, 8>;
	using int3 = Vec3<int, 4>;
	using int4 = Vec4<int, 16>;

	using char2 = Vec2<char, 2>;
	using char3 = Vec3<char, 1>;
	using char4 = Vec4<char, 4>;

	using short2 = Vec2<short, 4>;
	using short3 = Vec3<short, 2>;
	using short4 = Vec4<short, 8>;

	using float2 = Vec2<float, 8>;
	using float3 = Vec3<float, 4>;
	using float4 = Vec4<float, 16>;

	using double2 = Vec2<double, 16>;
	using double3 = Vec3<double,  8>;
	using double4 = Vec4<double, 16>;
	using double4_16a = Vec4<double, 16>;
	using double4_32a = Vec4<double, 32>;

	using uint = unsigned int;
	using uint2 = Vec2<unsigned int, 8>;
	using uint3 = Vec3<unsigned int, 4>;
	using uint4 = Vec4<unsigned int, 16>;

	using longlong = long long;
	using longlong2 = Vec2<long long, 16>;
	using longlong3 = Vec3<long long,  8>;
	using longlong4 = Vec4<long long, 16>;
	using longlong4_16a = Vec4<long long, 16>;
	using longlong4_32a = Vec4<long long, 32>;

	using uchar = unsigned char;
	using uchar2 = Vec2<unsigned char, 2>;
	using uchar3 = Vec3<unsigned char, 1>;
	using uchar4 = Vec4<unsigned char, 4>;

	using ushort = unsigned short;
	using ushort2 = Vec2<unsigned short, 4>;
	using ushort3 = Vec3<unsigned short, 2>;
	using ushort4 = Vec4<unsigned short, 8>;

	using ulonglong = unsigned long long;
	using ulonglong2 = Vec2<unsigned long long, 16>;
	using ulonglong3 = Vec3<unsigned long long,  8>;
	using ulonglong4 = Vec4<unsigned long long, 16>;
	using ulonglong4_16a = Vec4<unsigned long long, 16>;
	using ulonglong4_32a = Vec4<unsigned long long, 32>;
#else
	using int2 = ::int2;
	using int3 = ::int3;
	using int4 = ::int4;

	using uint = unsigned int;
	using uint2 = ::uint2;
	using uint3 = ::uint3;
	using uint4 = ::uint4;
	
	using char2 = ::char2;
	using char3 = ::char3;
	using char4 = ::char4;

	using uchar = unsigned char;
	using uchar2 = ::uchar2;
	using uchar3 = ::uchar3;
	using uchar4 = ::uchar4;

	using short2 = ::short2;
	using short3 = ::short3;
	using short4 = ::short4;

	using float2 = ::float2;
	using float3 = ::float3;
	using float4 = ::float4;

	using double2 = ::double2;
	using double3 = ::double3;
#if __CUDACC_VER_MAJOR__ >= 13
	using double4 = ::double4_16a;
	using double4_16a = ::double4_16a;
	using double4_32a = ::double4_32a;
#else
	using double4 = ::double4;
	using double4_16a = ::double4;
	using double4_32a = Vec4<double, 32>;
#endif

	using ushort = unsigned short;
	using ushort2 = ::ushort2;
	using ushort3 = ::ushort3;
	using ushort4 = ::ushort4;

	using longlong = long long;
	using longlong2 = ::longlong2;
	using longlong3 = ::longlong3;
#if __CUDACC_VER_MAJOR__ >= 13
	using longlong4 = ::longlong4_16a;
	using longlong4_16a = ::longlong4_16a;
	using longlong4_32a = ::longlong4_32a;
#else
	using longlong4 = ::longlong4;
	using longlong4_16a = ::longlong4;
	using longlong4_32a = Vec4<long long, 32>;
#endif

	using ulonglong = unsigned long long;
	using ulonglong2 = ::ulonglong2;
	using ulonglong3 = ::ulonglong3;
#if __CUDACC_VER_MAJOR__ >= 13
	using ulonglong4 = ::ulonglong4_16a;
	using ulonglong4_16a = ::ulonglong4_16a;
	using ulonglong4_32a = ::ulonglong4_32a;
#else
	using ulonglong4 = ::ulonglong4;
	using ulonglong4_16a = ::ulonglong4;
	using ulonglong4_32a = Vec4<unsigned long long, 32>;
#endif
#endif
}