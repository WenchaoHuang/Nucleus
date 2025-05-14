/**
 *	Copyright (c) 2025 Huang Wenchao <physhuangwenchao@gmail.com>
 *
 *	All rights reserved. Use of this source code is governed by a
 *	GPL-2.0 license that can be found in the LICENSE file.
 *
 *	This program is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *	GNU General Public License for more details.
 */
#pragma once

#include <cstdio>
#include <assert.h>

#pragma warning(disable: 4201)		//!	Nonstandard extension used: nameless struct/union.

/*************************************************************************
***********************    CUDA Compatibilities    ***********************
*************************************************************************/

#if defined(__CUDACC__)
	#define	NS_INLINE						__inline__
	#define NS_ALIGN(n)						__align__(n)
	#define NS_FORCE_INLINE					__forceinline__
	#define	NS_CUDA_CALLABLE				__host__ __device__
	#define	NS_CUDA_CALLABLE_INLINE			__host__ __device__ __inline__
#else
	#define	NS_INLINE						inline
	#define NS_ALIGN(n)						alignas(n)
	#define NS_FORCE_INLINE					__forceinline
	#define	NS_CUDA_CALLABLE
	#define	NS_CUDA_CALLABLE_INLINE			inline

	#ifndef __device__
		#define __device__
	#endif
#endif

#if defined(__CUDA_ARCH__)
	#define	NS_ASSERT(expression)
#else
	#define	NS_ASSERT(expression)			assert(expression)
#endif

/*************************************************************************
***************************    Noncopyable    ****************************
*************************************************************************/

#define NS_NONCOPYABLE(ClassName)										\
																		\
	ClassName(const ClassName&) = delete;								\
																		\
	ClassName & operator=(const ClassName&) = delete;					\

/*************************************************************************
****************************    Namespace    *****************************
*************************************************************************/

#ifndef NS_NAMESPACE
	#define NS_NAMESPACE				ns
#endif

#define NS_USING_NAMESPACE				using namespace NS_NAMESPACE;

namespace NS_NAMESPACE {}