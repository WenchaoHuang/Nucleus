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

#include "macros.hpp"

namespace NS_NAMESPACE
{
	/*********************************************************************
	***************************    Version    ****************************
	*********************************************************************/

	/**
	 *	@brief		CUDA version number.
	 */
	struct Version
	{
		union
		{
			struct { int Minor, Major; };
			struct { long long Encoded; };
		};

		//	Constructors
		constexpr Version() : Major(0), Minor(0) {}
		constexpr Version(int major, int minor) : Major(major), Minor(minor) {}

		//	Compare operators
		constexpr bool operator==(Version rhs) const { return Encoded == rhs.Encoded; }
		constexpr bool operator!=(Version rhs) const { return Encoded != rhs.Encoded; }
		constexpr bool operator<=(Version rhs) const { return Encoded <= rhs.Encoded; }
		constexpr bool operator>=(Version rhs) const { return Encoded >= rhs.Encoded; }
		constexpr bool operator<(Version rhs) const { return Encoded < rhs.Encoded; }
		constexpr bool operator>(Version rhs) const { return Encoded > rhs.Encoded; }
	};
}