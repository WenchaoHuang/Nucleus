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

#include <memory>
#include "macros.h"

/*************************************************************************
***********************    Forward Declarations    ***********************
*************************************************************************/

#ifndef __CUDACC__
	struct dim3;
	struct cudaDeviceProp;
	typedef enum cudaError : int				cudaError_t;
	typedef struct cudaArray *					cudaArray_t;
	typedef struct CUevent_st *					cudaEvent_t;
	typedef struct CUgraph_st *					cudaGraph_t;
	typedef struct CUstream_st *				cudaStream_t;
	typedef struct CUgraphNode_st *				cudaGraphNode_t;
	typedef struct CUgraphExec_st *				cudaGraphExec_t;
	typedef struct cudaMipmappedArray *			cudaMipmappedArray_t;
	typedef unsigned long long					cudaTextureObject_t;
	typedef unsigned long long					cudaSurfaceObject_t;
#endif

namespace NS_NAMESPACE
{
	class Event;
	class Graph;
	class Buffer;
	class Stream;
	class Device;
	class Context;
	class Texture;
	class Allocator;
	class TimedEvent;
	class ScopedTimer;
	class HostAllocator;
	class DeviceAllocator;

	struct Version;
	struct Sampler;
	enum class Format;
	enum class Result;
	enum class FilterMode;
	enum class AddressMode;

	//	For device objects.
	namespace dev
	{
		template<typename Type> struct Ptr;
		template<typename Type> struct Ptr2;
		template<typename Type> struct Ptr3;

		template<typename Type> struct Surf1D;
		template<typename Type> struct Surf2D;
		template<typename Type> struct Surf3D;
		template<typename Type> struct SurfCube;
		template<typename Type> struct Surf1DLayered;
		template<typename Type> struct Surf2DLayered;
		template<typename Type> struct SurfCubeLayered;

		template<typename Type> struct Tex1D;
		template<typename Type> struct Tex2D;
		template<typename Type> struct Tex3D;
		template<typename Type> struct TexCube;
		template<typename Type> struct Tex1DLod;
		template<typename Type> struct Tex2DLod;
		template<typename Type> struct Tex3DLod;
		template<typename Type> struct TexCubeLod;
		template<typename Type> struct Tex1DLayered;
		template<typename Type> struct Tex2DLayered;
		template<typename Type> struct TexCubeLayered;
		template<typename Type> struct Tex1DLayeredLod;
		template<typename Type> struct Tex2DLayeredLod;
		template<typename Type> struct TexCubeLayeredLod;
	}

	template<typename Type> class Array;
	template<typename Type> class Array2D;
	template<typename Type> class Array3D;

	template<typename Type> class BufferView1D;
	template<typename Type> class BufferView2D;
	template<typename Type> class BufferView3D;

	template<typename Type> class Image1D;
	template<typename Type> class Image2D;
	template<typename Type> class Image3D;
	template<typename Type> class ImageCube;
	template<typename Type> class Image1DLayered;
	template<typename Type> class Image2DLayered;
	template<typename Type> class ImageCubeLayered;
	template<typename Type> class Image1DLod;
	template<typename Type> class Image2DLod;
	template<typename Type> class Image3DLod;
	template<typename Type> class ImageCubeLod;
	template<typename Type> class Image1DLayeredLod;
	template<typename Type> class Image2DLayeredLod;
	template<typename Type> class ImageCubeLayeredLod;

	template<typename Type> class Surface1D;
	template<typename Type> class Surface2D;
	template<typename Type> class Surface3D;
	template<typename Type> class SurfaceCube;
	template<typename Type> class Surface1DLayered;
	template<typename Type> class Surface2DLayered;
	template<typename Type> class SurfaceCubeLayered;

	template<typename Type> class Texture1D;
	template<typename Type> class Texture2D;
	template<typename Type> class Texture3D;
	template<typename Type> class TextureCube;
	template<typename Type> class Texture1DLod;
	template<typename Type> class Texture2DLod;
	template<typename Type> class Texture3DLod;
	template<typename Type> class TextureCubeLod;
	template<typename Type> class Texture1DLayered;
	template<typename Type> class Texture2DLayered;
	template<typename Type> class TextureCubeLayered;
	template<typename Type> class Texture1DLayeredLod;
	template<typename Type> class Texture2DLayeredLod;
	template<typename Type> class TextureCubeLayeredLod;

	template<typename Type> struct ImageAccessor;
	template<typename... Args> using KernelFunc = void(*)(Args...);

	using BufferPtr			= std::shared_ptr<Buffer>;
	using AllocPtr			= std::shared_ptr<Allocator>;
	using HostAllocPtr		= std::shared_ptr<HostAllocator>;
	using DevAllocPtr		= std::shared_ptr<DeviceAllocator>;
}

/*************************************************************************
****************************    Type Alias    ****************************
*************************************************************************/

using NsEvent											= NS_NAMESPACE::Event;
using NsGraph											= NS_NAMESPACE::Graph;
using NsBuffer											= NS_NAMESPACE::Buffer;
using NsStream											= NS_NAMESPACE::Stream;
using NsDevice											= NS_NAMESPACE::Device;
using NsFormat											= NS_NAMESPACE::Format;
using NsSampler											= NS_NAMESPACE::Sampler;
using NsContext											= NS_NAMESPACE::Context;
using NsVersion											= NS_NAMESPACE::Version;
using NsFilterMode										= NS_NAMESPACE::FilterMode;
using NsTimedEvent										= NS_NAMESPACE::TimedEvent;
using NsScopedTimer										= NS_NAMESPACE::ScopedTimer;
using NsAddressMode										= NS_NAMESPACE::AddressMode;
using NsHostAlloc										= NS_NAMESPACE::HostAllocator;
using NsDevAlloc										= NS_NAMESPACE::DeviceAllocator;
using NsAlloc											= NS_NAMESPACE::Allocator;

using NsBufferPtr										= NS_NAMESPACE::BufferPtr;
using NsHostAllocPtr									= NS_NAMESPACE::HostAllocPtr;
using NsDevAllocPtr										= NS_NAMESPACE::DevAllocPtr;
using NsAllocPtr										= NS_NAMESPACE::AllocPtr;

template<typename Type> using NsArray					= NS_NAMESPACE::Array<Type>;
template<typename Type> using NsArray2D					= NS_NAMESPACE::Array2D<Type>;
template<typename Type> using NsArray3D					= NS_NAMESPACE::Array3D<Type>;

template<typename Type> using NsBufferView1D			= NS_NAMESPACE::BufferView1D<Type>;
template<typename Type> using NsBufferView2D			= NS_NAMESPACE::BufferView2D<Type>;
template<typename Type> using NsBufferView3D			= NS_NAMESPACE::BufferView3D<Type>;

template<typename Type> using NsImage1D					= NS_NAMESPACE::Image1D<Type>;
template<typename Type> using NsImage2D					= NS_NAMESPACE::Image2D<Type>;
template<typename Type> using NsImage3D					= NS_NAMESPACE::Image3D<Type>;
template<typename Type> using NsImageCube				= NS_NAMESPACE::ImageCube<Type>;
template<typename Type> using NsImage1DLayered			= NS_NAMESPACE::Image1DLayered<Type>;
template<typename Type> using NsImage2DLayered			= NS_NAMESPACE::Image2DLayered<Type>;
template<typename Type> using NsImageCubeLayered		= NS_NAMESPACE::ImageCubeLayered<Type>;

template<typename Type> using NsImage1DLod				= NS_NAMESPACE::Image1DLod<Type>;
template<typename Type> using NsImage2DLod				= NS_NAMESPACE::Image2DLod<Type>;
template<typename Type> using NsImage3DLod				= NS_NAMESPACE::Image3DLod<Type>;
template<typename Type> using NsImageCubeLod			= NS_NAMESPACE::ImageCubeLod<Type>;
template<typename Type> using NsImage1DLayeredLod		= NS_NAMESPACE::Image1DLayeredLod<Type>;
template<typename Type> using NsImage2DLayeredLod		= NS_NAMESPACE::Image2DLayeredLod<Type>;
template<typename Type> using NsImageCubeLayeredLod		= NS_NAMESPACE::ImageCubeLayeredLod<Type>;

template<typename Type> using NsSurf1D					= NS_NAMESPACE::Surface1D<Type>;
template<typename Type> using NsSurf2D					= NS_NAMESPACE::Surface2D<Type>;
template<typename Type> using NsSurf3D					= NS_NAMESPACE::Surface3D<Type>;
template<typename Type> using NsSurfCube				= NS_NAMESPACE::SurfaceCube<Type>;
template<typename Type> using NsSurf1DLayered			= NS_NAMESPACE::Surface1DLayered<Type>;
template<typename Type> using NsSurf2DLayered			= NS_NAMESPACE::Surface2DLayered<Type>;
template<typename Type> using NsSurfCubeLayered			= NS_NAMESPACE::SurfaceCubeLayered<Type>;

template<typename Type> using NsTex1D					= NS_NAMESPACE::Texture1D<Type>;
template<typename Type> using NsTex2D					= NS_NAMESPACE::Texture2D<Type>;
template<typename Type> using NsTex3D					= NS_NAMESPACE::Texture3D<Type>;
template<typename Type> using NsTexCube					= NS_NAMESPACE::TextureCube<Type>;
template<typename Type> using NsTex1DLod				= NS_NAMESPACE::Texture1DLod<Type>;
template<typename Type> using NsTex2DLod				= NS_NAMESPACE::Texture2DLod<Type>;
template<typename Type> using NsTex3DLod				= NS_NAMESPACE::Texture3DLod<Type>;
template<typename Type> using NsTexCubeLod				= NS_NAMESPACE::TextureCubeLod<Type>;
template<typename Type> using NsTex1DLayered			= NS_NAMESPACE::Texture1DLayered<Type>;
template<typename Type> using NsTex2DLayered			= NS_NAMESPACE::Texture2DLayered<Type>;
template<typename Type> using NsTexCubeLayered			= NS_NAMESPACE::TextureCubeLayered<Type>;
template<typename Type> using NsTex1DLayeredLod			= NS_NAMESPACE::Texture1DLayeredLod<Type>;
template<typename Type> using NsTex2DLayeredLod			= NS_NAMESPACE::Texture2DLayeredLod<Type>;
template<typename Type> using NsTexCubeLayeredLod		= NS_NAMESPACE::TextureCubeLayeredLod<Type>;