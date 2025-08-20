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

#include "image_1d.h"
#include "image_2d.h"
#include "image_3d.h"
#include "image_cube.h"
#include "device_surface.h"

namespace NS_NAMESPACE
{
	/*****************************************************************************
	*******************************    Surface    ********************************
	*****************************************************************************/

	class Surface	//	Base class to manage CUDA surface resources.
	{
		NS_NONCOPYABLE(Surface)

	public:

		//	Default constructor
		NS_API Surface();

		//	Destructor
		NS_API ~Surface();

	public:

		//	Unbinds the current surface resource.
		NS_API void unbind();

		//	Checks if the surface is empty.
		bool empty() const { return m_hSurface == 0; }

		//	Return CUDA type of this object.
		cudaSurfaceObject_t handle() const { return m_hSurface; }

	protected:

		/**
		 *	@brief		Binds a texture memory as the surface resource.
		 *	@param[in]	image - Shared pointer to the memory object.
		 *	@throw		cudaError_t - In case of failure.
		 */
		NS_API void bindImage(std::shared_ptr<Image> image);

	protected:

		cudaSurfaceObject_t			m_hSurface;

		std::shared_ptr<Image>		m_image;
	};

	/*****************************************************************************
	******************************    Surface1D    *******************************
	*****************************************************************************/

	//	Representing a CUDA 1D surface object.
	template<typename Type> class Surface1D : public Surface
	{

	public:

		//	Default constructor.
		Surface1D() {}

		//	Constructs a 1D surface and binds a texture memory object.
		explicit Surface1D(std::shared_ptr<Image1D<Type>> image) { this->bind(image); }

	public:

		//	Binds a 1D texture memory object to the surface.
		void bind(std::shared_ptr<Image1D<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<Image1D<Type>> image() const { return std::dynamic_pointer_cast<Image1D<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::Surf1D<Type>() { return m_image ? dev::Surf1D<Type>(m_hSurface, m_image->width()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::Surf1D<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::Surf1D<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::Surf1D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	******************************    Surface2D    *******************************
	*****************************************************************************/

	//	Representing a CUDA 2D surface object.
	template<typename Type> class Surface2D : public Surface
	{

	public:

		//	Default constructor.
		Surface2D() {}

		//	Constructs a 2D surface and binds a texture memory object.
		explicit Surface2D(std::shared_ptr<Image2D<Type>> image) { this->bind(image); }

	public:

		//	Binds a 2D texture memory object to the surface.
		void bind(std::shared_ptr<Image2D<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<Image2D<Type>> image() const { return std::dynamic_pointer_cast<Image2D<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::Surf2D<Type>() { return m_image ? dev::Surf2D<Type>(m_hSurface, image()->width(), image()->height()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::Surf2D<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::Surf2D<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::Surf2D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	******************************    Surface3D    *******************************
	*****************************************************************************/

	//	Representing a CUDA 3D surface object.
	template<typename Type> class Surface3D : public Surface
	{

	public:

		//	Default constructor.
		Surface3D() {}

		//	Constructs a 3D surface and binds a texture memory object.
		explicit Surface3D(std::shared_ptr<Image3D<Type>> image) { this->bind(image); }

	public:

		//	Binds a 3D texture memory object to the surface.
		void bind(std::shared_ptr<Image3D<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<Image3D<Type>> image() const { return std::dynamic_pointer_cast<Image3D<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::Surf3D<Type>() { return m_image ? dev::Surf3D<Type>(m_hSurface, image()->width(), image()->height(), image()->depth()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::Surf3D<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::Surf3D<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::Surf3D<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	***************************    Surface1DLayered    ***************************
	*****************************************************************************/

	//	Representing a CUDA 1D layered surface object.
	template<typename Type> class Surface1DLayered : public Surface
	{

	public:

		//	Default constructor.
		Surface1DLayered() {}

		//	Constructs a 1D layered surface and binds a texture memory object.
		explicit Surface1DLayered(std::shared_ptr<Image1DLayered<Type>> image) { this->bind(image); }

	public:

		//	Binds a 1D layered texture memory object to the surface.
		void bind(std::shared_ptr<Image1DLayered<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<Image1DLayered<Type>> image() const { return std::dynamic_pointer_cast<Image1DLayered<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::Surf1DLayered<Type>() { return m_image ? dev::Surf1DLayered<Type>(m_hSurface, image()->width(), image()->numLayers()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::Surf1DLayered<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::Surf1DLayered<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::Surf1DLayered<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	***************************    Surface2DLayered    ***************************
	*****************************************************************************/

	//	Representing a CUDA 2D layered surface object.
	template<typename Type> class Surface2DLayered : public Surface
	{

	public:

		//	Default constructor.
		Surface2DLayered() {}

		//	Constructs a 2D layered surface and binds a texture memory object.
		explicit Surface2DLayered(std::shared_ptr<Image2DLayered<Type>> image) { this->bind(image); }

	public:

		//	Binds a 2D layered texture memory object to the surface.
		void bind(std::shared_ptr<Image2DLayered<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<Image2DLayered<Type>> image() const { return std::dynamic_pointer_cast<Image2DLayered<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::Surf2DLayered<Type>() { return m_image ? dev::Surf2DLayered<Type>(m_hSurface, image()->width(), image()->height(), image()->numLayers()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::Surf2DLayered<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::Surf2DLayered<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::Surf2DLayered<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	*****************************    SurfaceCube    ******************************
	*****************************************************************************/

	//	Representing a CUDA cubemap surface object.
	template<typename Type> class SurfaceCube : public Surface
	{

	public:

		//	Default constructor.
		SurfaceCube() {}

		//	Constructs a cubemap surface and binds a texture memory object.
		explicit SurfaceCube(std::shared_ptr<ImageCube<Type>> image) { this->bind(image); }

	public:

		//	Binds a cubemap texture memory object to the surface.
		void bind(std::shared_ptr<ImageCube<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<ImageCube<Type>> image() const { return std::dynamic_pointer_cast<ImageCube<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::SurfCube<Type>() { return m_image ? dev::SurfCube<Type>(m_hSurface, image()->width()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::SurfCube<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::SurfCube<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::SurfCube<Type> accessor() { return *this; }
	};

	/*****************************************************************************
	**************************    SurfaceCubeLayered    **************************
	*****************************************************************************/

	//	Representing a CUDA layered cubemap surface object.
	template<typename Type> class SurfaceCubeLayered : public Surface
	{

	public:

		//	Default constructor.
		SurfaceCubeLayered() {}

		//	Constructs a layered cubemap surface and binds a texture memory object.
		explicit SurfaceCubeLayered(std::shared_ptr<ImageCubeLayered<Type>> image) { this->bind(image); }

	public:

		//	Binds a layered cubemap texture memory object to the surface.
		void bind(std::shared_ptr<ImageCubeLayered<Type>> image) { this->bindImage(image); }

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<ImageCubeLayered<Type>> image() const { return std::dynamic_pointer_cast<ImageCubeLayered<Type>>(m_image); }

		//	Converts to a device surface object for kernal access.
		operator dev::SurfCubeLayered<Type>() { return m_image ? dev::SurfCubeLayered<Type>(m_hSurface, image()->width(), image()->numLayers()) : nullptr; }

		//	Converts to a device surface object for kernal access (constant version).
		operator dev::SurfCubeLayered<const Type>() const { return *this; }

		//	Returns read-only device accessor explicitly.
		dev::SurfCubeLayered<const Type> accessor() const { return *this; }

		//	Returns device accessor explicitly.
		dev::SurfCubeLayered<Type> accessor() { return *this; }
	};
}