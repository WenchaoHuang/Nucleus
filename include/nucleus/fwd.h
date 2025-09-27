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

/*********************************************************************************
***************************    Forward Declarations    ***************************
*********************************************************************************/

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

	//	Trait to check if two types are binary compatible in terms of size and alignment.
	template<typename Type1, typename Type2> struct BinaryCompatible
	{
		static constexpr bool value = (sizeof(Type1) == sizeof(Type2)) && (alignof(Type1) == alignof(Type2));
	};

	//	Utility functions to reinterpret buffer views as another compatible element type.
	template<typename DstType, typename SrcType> BufferView1D<DstType> view_cast(BufferView1D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<DstType> view_cast(BufferView2D<SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<DstType> view_cast(BufferView3D<SrcType> view);

	template<typename DstType, typename SrcType> BufferView1D<const DstType> view_cast(BufferView1D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView2D<const DstType> view_cast(BufferView2D<const SrcType> view);
	template<typename DstType, typename SrcType> BufferView3D<const DstType> view_cast(BufferView3D<const SrcType> view);

	// Utility functions to reinterpret device pointers as another compatible element type.
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<const DstType> ptr_cast(dev::Ptr<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<const DstType> ptr_cast(dev::Ptr2<const SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<const DstType> ptr_cast(dev::Ptr3<const SrcType> ptr);

	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr<DstType> ptr_cast(dev::Ptr<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr2<DstType> ptr_cast(dev::Ptr2<SrcType> ptr);
	template<typename DstType, typename SrcType> NS_CUDA_CALLABLE dev::Ptr3<DstType> ptr_cast(dev::Ptr3<SrcType> ptr);
}