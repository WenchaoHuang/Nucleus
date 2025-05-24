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

#include "format.h"

#ifndef __CUDACC__
	enum cudaSurfaceBoundaryMode;
#else
	#include <surface_indirect_functions.h>
#endif

namespace NS_NAMESPACE
{
	//	Defines the internal CUDA-compatible value type for a given C++ type.
	template<typename Type> using internal_value_type = typename details::TypeMapping<FormatMapping<std::remove_const_t<Type>>::value>::type;

	/*********************************************************************
	**************************    devSurface    **************************
	*********************************************************************/

	/**
	 *	@brief		Base class for all CUDA surface objects.
	 *	@details	Provides empy constructor for compatibility with CUDA constant memory.
	 *	@details	This struct encapsulates a CUDA surface object handle and provides
	 *				common interface methods for derived surface types. It allows querying
	 *				the underlying CUDA surface handle, checking if the surface is valid or empty,
	 *				and supports implicit boolean conversion for validity checks.
	 */
	struct devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurface() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurface(std::nullptr_t) : m_hSurface(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurface(cudaSurfaceObject_t hSurface) : m_hSurface(hSurface) {}

		//	Return CUDA type of this object for compatibility.
		NS_CUDA_CALLABLE cudaSurfaceObject_t getHandle() const { return m_hSurface; }

		//	Bool conversion operator.
		NS_CUDA_CALLABLE operator bool() const { return m_hSurface != 0; }

		//	Check if the surface is empty.
		NS_CUDA_CALLABLE bool empty() const { return m_hSurface == 0; }

	protected:

		cudaSurfaceObject_t		m_hSurface;
	};

	/*********************************************************************
	********************    devSurf1D<const Type>    *********************
	*********************************************************************/

	/**
	 *	@brief		Represents a 1D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf1D<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf1D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf1D(std::nullptr_t) : devSurface(nullptr), m_width(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf1D(cudaSurfaceObject_t hSurface, uint32_t width) : devSurface(hSurface), m_width(width) {}

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surf1Dread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
	};

	/*********************************************************************
	***********************    devSurf1D<Type>    ************************
	*********************************************************************/

	/**
	 *	@brief		Represents a 1D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf1D : public devSurf1D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf1D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf1D(std::nullptr_t) : devSurf1D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf1D(cudaSurfaceObject_t hSurface, uint32_t width) : devSurf1D<const Type>(hSurface, width) {}
		
		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf1Dwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	********************    devSurf2D<const Type>    *********************
	*********************************************************************/

	/**
	 *	@brief		Represents a 2D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf2D<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf2D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf2D(std::nullptr_t) : devSurface(nullptr), m_width(0), m_height(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf2D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height) : devSurface(hSurface), m_width(width), m_height(height) {}

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surf2Dread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, y, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
	};

	/*********************************************************************
	***********************    devSurf2D<Type>    ************************
	*********************************************************************/

	/**
	 *	@brief		Represents a 2D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf2D : public devSurf2D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf2D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf2D(std::nullptr_t) : devSurf2D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf2D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height) : devSurf2D<const Type>(hSurface, width, height) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf2Dwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, y, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	********************    devSurf3D<const Type>    *********************
	*********************************************************************/

	/**
	 *	@brief		Represents a 3D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 3D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf3D<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf3D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf3D(std::nullptr_t) : devSurface(nullptr), m_width(0), m_height(0), m_depth(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf3D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t depth) : devSurface(hSurface), m_width(width), m_height(height), m_depth(depth) {}

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return depth of the buffer.
		NS_CUDA_CALLABLE uint32_t depth() const { return m_depth; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surf3Dread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, y, z, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
		uint32_t		m_depth;
	};

	/*********************************************************************
	***********************    devSurf3D<Type>    ************************
	*********************************************************************/

	/**
	 *	@brief		Represents a 3D CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 3D CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf3D : public devSurf3D<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf3D() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf3D(std::nullptr_t) : devSurf3D<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf3D(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t depth) : devSurf3D<const Type>(hSurface, width, height, depth) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf3Dwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, y, z, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	*****************    devSurf1DLayered<const Type>    *****************
	*********************************************************************/

	/**
	 *	@brief		Represents a 1D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf1DLayered<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf1DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf1DLayered(std::nullptr_t) : devSurface(nullptr), m_width(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf1DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : devSurface(hSurface), m_width(width), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surf1DLayeredread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, layer, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_numLayers;
	};

	/*********************************************************************
	********************    devSurf1DLayered<Type>    ********************
	*********************************************************************/

	/**
	 *	@brief		Represents a 1D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 1D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf1DLayered : public devSurf1DLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf1DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf1DLayered(std::nullptr_t) : devSurf1DLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf1DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : devSurf1DLayered<const Type>(hSurface, width, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf1DLayeredwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, layer, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	*****************    devSurf2DLayered<const Type>    *****************
	*********************************************************************/

	/**
	 *	@brief		Represents a 2D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf2DLayered<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf2DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf2DLayered(std::nullptr_t) : devSurface(nullptr), m_width(0), m_height(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf2DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t numLayers) : devSurface(hSurface), m_width(width), m_height(height), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return height of the buffer.
		NS_CUDA_CALLABLE uint32_t height() const { return m_height; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surf2DLayeredread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, y, layer, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_height;
		uint32_t		m_numLayers;
	};

	/*********************************************************************
	********************    devSurf2DLayered<Type>    ********************
	*********************************************************************/

	/**
	 *	@brief		Represents a 2D layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for 2D layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurf2DLayered : public devSurf2DLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurf2DLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurf2DLayered(std::nullptr_t) : devSurf2DLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurf2DLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t height, uint32_t numLayers) : devSurf2DLayered<const Type>(hSurface, width, height, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surf2DLayeredwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, y, layer, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	*******************    devSurfCube<const Type>    ********************
	*********************************************************************/
	
	/**
	 *	@brief		Represents a cube-type CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurfCube<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurfCube() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurfCube(std::nullptr_t) : devSurface(nullptr), m_width(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurfCube(cudaSurfaceObject_t hSurface, uint32_t width) : devSurface(hSurface), m_width(width) {}

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surfCubemapread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, y, face, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
	};

	/*********************************************************************
	**********************    devSurfCube<Type>    ***********************
	*********************************************************************/

	/**
	 *	@brief		Represents a cube-type CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurfCube : public devSurfCube<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurfCube() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurfCube(std::nullptr_t) : devSurfCube<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurfCube(cudaSurfaceObject_t hSurface, uint32_t width) : devSurfCube<const Type>(hSurface, width) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surfCubemapwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, y, face, boundaryMode);
		}
	#endif
	};

	/*********************************************************************
	****************    devSurfCubeLayered<const Type>    ****************
	*********************************************************************/

	/**
	 *	@brief		Represents a cube-type layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurfCubeLayered<const Type> : public devSurface
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurfCubeLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurfCubeLayered(std::nullptr_t) : devSurface(nullptr), m_width(0), m_numLayers(0) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurfCubeLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : devSurface(hSurface), m_width(width), m_numLayers(numLayers) {}

		//	Return the number of layers.
		NS_CUDA_CALLABLE uint32_t numLayers() const { return m_numLayers; }

		//	Return width of the buffer.
		NS_CUDA_CALLABLE uint32_t width() const { return m_width; }

		//	Read method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ Type read(int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ Type read(int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			internal_value_type<Type> value;

			surfCubemapLayeredread<decltype(value)>(&value, m_hSurface, sizeof(Type) * x, y, 6 * layer + face, boundaryMode);

			return reinterpret_cast<Type&>(value);
		}
	#endif

	protected:

		uint32_t		m_width;
		uint32_t		m_numLayers;
	};

	/*********************************************************************
	*******************    devSurfCubeLayered<Type>    *******************
	*********************************************************************/

	/**
	 *	@brief		Represents a cube-type layered CUDA surface object for device access.
	 *	@tparam		Type - The data type stored in the surface.
	 *	@details	This struct provides an interface for cube-type layered CUDA surface objects,
	 *				enabling device-side read and write operations (read-only for const Type).
	 */
	template<typename Type> struct devSurfCubeLayered : public devSurfCubeLayered<const Type>
	{
		//	Default constructor.
		NS_CUDA_CALLABLE devSurfCubeLayered() {}

		//	Constructor with nullptr.
		NS_CUDA_CALLABLE devSurfCubeLayered(std::nullptr_t) : devSurfCubeLayered<const Type>(nullptr) {}

		//	Constructor with cudaSurfaceObject_t.
		NS_CUDA_CALLABLE explicit devSurfCubeLayered(cudaSurfaceObject_t hSurface, uint32_t width, uint32_t numLayers) : devSurfCubeLayered<const Type>(hSurface, width, numLayers) {}

		//	Write method for CUDA surface object.
	#ifndef __CUDACC__
		__device__ void write(Type value, int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = 2) const;
	#else
		__device__ void write(Type value, int x, int y, int face, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) const
		{
			surfCubemapLayeredwrite<internal_value_type<Type>>(reinterpret_cast<internal_value_type<Type>&>(value), m_hSurface, sizeof(Type) * x, y, 6 * layer + face, boundaryMode);
		}
	#endif
	};
}