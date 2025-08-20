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
#include "format.h"
#include "sampler.h"
#include "device_texture.h"

namespace NS_NAMESPACE
{
	class ImageBase;

	/*****************************************************************************
	*******************************    Texture    ********************************
	*****************************************************************************/

	class Texture	//	Base class to manage CUDA texture resources.
	{
		NS_NONCOPYABLE(Texture)

	protected:

		//	Default constructor
		NS_API Texture();

		//	Destructor
		NS_API ~Texture();

	public:

		//	Unbinds the current surface resource.
		NS_API void unbind() noexcept;

		//	Checks if the surface is empty.
		bool empty() const { return m_hTexture == 0; }

		//	Returns sampler of the texture object.
		const Sampler & sampler() const { return m_sampler; }

	protected:

		/**
		 *	@brief		Binds a texture memory as the texture resource.
		 *	@param[in]	image - Shared pointer to the image object.
		 *	@param[in]	sampler - Sampler for texture fetched.
		 *	@param[in]	viewFormat - View format of texture (internal use).
		 *	@throws		cudaError_t - In case of failure.
		 */
		NS_API void bindImage(std::shared_ptr<ImageBase> image, Sampler sampler, Format viewFormat);

	protected:

		std::shared_ptr<ImageBase>		m_image;
		cudaTextureObject_t				m_hTexture;
		Sampler							m_sampler;
	};

	/*****************************************************************************
	************************    details::_Impl_Texture    ************************
	*****************************************************************************/

	namespace details
	{
		//	Internal texture template for textures with floating-type texel format.
		template<template<typename> class ImageTemplate, template<typename> class devTexTemplate, typename Type> class _Impl_Texture : public Texture
		{

		public:

			/**
			 *	@brief		Binds a texture memory as the texture resource.
			 *	@param[in]	image - Shared pointer to the memory object.
			 *	@param[in]	sampler - Sampler for texture fetched.
			 *	@throws		cudaError_t - In case of failure.
			 */
			void bind(std::shared_ptr<ImageTemplate<Type>> image, Sampler sampler = Sampler()) { this->bindImage(image, sampler, ns::FormatMapping<Type>::value); }


			/**
			 *	@brief		Binds a texture memory as the texture resource.
			 *	@param[in]	image - Shared pointer to the memory object.
			 *	@param[in]	sampler - Sampler for texture fetched.
			 *	@details	Force set ReadMode::eNormalizedFloat to 1.
			 *	@throws		cudaError_t - In case of failure.
			 */
			template<typename StorageType> void bind(std::shared_ptr<ImageTemplate<StorageType>> pImage, Sampler sampler = Sampler())
			{
				//	Validate that source and destination formats have the same number of components.
				static_assert(FormatTraits<ns::FormatMapping<StorageType>::value>::component_count == FormatTraits<ns::FormatMapping<Type>::value>::component_count,
							  "Source and destination formats must have the same component count.");

				//	Prohibit unsigned integer source formats.
				static_assert(!std::is_same_v<typename FormatTraits<ns::FormatMapping<StorageType>::value>::component_type, unsigned int>,
							  "Unsigned integer source formats are not supported.");

				//	Prohibit signed integer source formats.
				static_assert(!std::is_same_v<typename FormatTraits<ns::FormatMapping<StorageType>::value>::component_type, int>,
							  "Signed integer source formats are not supported.");

				//	Enforce float destination format.
				static_assert(std::is_same_v<typename FormatTraits<ns::FormatMapping<Type>::value>::component_type, float>,
							  "Destination format must use float components.");

				//	Perform the actual texture binding operation
				this->bindImage(pImage, sampler, ns::FormatMapping<Type>::value);
			}

		public:

			//	Returns shared pointer to the binded texture memory.
			std::shared_ptr<ImageTemplate<void>> image() const { return std::dynamic_pointer_cast<ImageTemplate<void>>(m_image); }

			//	Converts to a device texture object for kernal access.
			operator devTexTemplate<Type>() const { return devTexTemplate<Type>(m_hTexture); }

			//	Returns device accessor explicitly.
			devTexTemplate<Type> accessor() const { return devTexTemplate<Type>(m_hTexture); }
		};
	}

	/*****************************************************************************
	*******************************    Texture    ********************************
	*****************************************************************************/

	template<typename Type> class Texture1D : public details::_Impl_Texture<Image1D, dev::Tex1D, Type> {};
	template<typename Type> class Texture2D : public details::_Impl_Texture<Image2D, dev::Tex2D, Type> {};
	template<typename Type> class Texture3D : public details::_Impl_Texture<Image3D, dev::Tex3D, Type> {};
	template<typename Type> class TextureCube : public details::_Impl_Texture<ImageCube, dev::TexCube, Type> {};
	template<typename Type> class Texture1DLod : public details::_Impl_Texture<Image1DLod, dev::Tex1DLod, Type> {};
	template<typename Type> class Texture2DLod : public details::_Impl_Texture<Image2DLod, dev::Tex2DLod, Type> {};
	template<typename Type> class Texture3DLod : public details::_Impl_Texture<Image3DLod, dev::Tex3DLod, Type> {};
	template<typename Type> class TextureCubeLod : public details::_Impl_Texture<ImageCubeLod, dev::TexCubeLod, Type> {};
	template<typename Type> class Texture1DLayered : public details::_Impl_Texture<Image1DLayered, dev::Tex1DLayered, Type> {};
	template<typename Type> class Texture2DLayered : public details::_Impl_Texture<Image2DLayered, dev::Tex2DLayered, Type> {};
	template<typename Type> class TextureCubeLayered : public details::_Impl_Texture<ImageCubeLayered, dev::TexCubeLayered, Type> {};
	template<typename Type> class Texture1DLayeredLod : public details::_Impl_Texture<Image1DLayeredLod, dev::Tex1DLayeredLod, Type> {};
	template<typename Type> class Texture2DLayeredLod : public details::_Impl_Texture<Image2DLayeredLod, dev::Tex2DLayeredLod, Type> {};
	template<typename Type> class TextureCubeLayeredLod : public details::_Impl_Texture<ImageCubeLayeredLod, dev::TexCubeLayeredLod, Type> {};
}