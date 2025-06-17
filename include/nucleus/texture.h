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

	/*********************************************************************
	***************************    Texture    ****************************
	*********************************************************************/

	class Texture	//	Base class to manage CUDA texture resources.
	{
		NS_NONCOPYABLE(Texture)

	protected:

		//	Default constructor
		Texture();

		//	Destructor
		~Texture();

	public:

		//	Unbinds the current surface resource.
		void unbind() noexcept;

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
		void bindImage(std::shared_ptr<ImageBase> image, Sampler sampler, Format viewFormat);

	protected:

		std::shared_ptr<ImageBase>		m_image;
		cudaTextureObject_t				m_hTexture;
		Sampler							m_sampler;
	};

namespace details
{
	/*********************************************************************
	***********************    IsFloatingFormat    ***********************
	*********************************************************************/

	template<typename Type> struct IsFloatingFormat
	{
		static constexpr bool value = (FormatMapping<Type>::value == Format::Float) || (FormatMapping<Type>::value == Format::Float2) || (FormatMapping<Type>::value == Format::Float4);
	};

	template<template<typename> class ImageTemplate, template<typename> class devTexTemplate, typename Type, bool IsFloatingType> class TextureBase;

	/*********************************************************************
	*************************    TextureBase    **************************
	*********************************************************************/

	//	Internal texture template for non-floating-type textures.
	template<template<typename> class ImageTemplate, template<typename> class devTexTemplate, typename Type> class TextureBase<ImageTemplate, devTexTemplate, Type, false> : public Texture
	{

	public:

		/**
		 *	@brief		Binds a texture memory as the texture resource.
		 *	@param[in]	image - Shared pointer to the memory object.
		 *	@param[in]	sampler - Sampler for texture fetched.
		 *	@throws		cudaError_t - In case of failure.
		 */
		void bind(std::shared_ptr<ImageTemplate<Type>> image, Sampler sampler = Sampler()) { this->bindImage(image, sampler, FormatMapping<Type>::value); }

	public:

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<ImageTemplate<Type>> getImage() const { return std::dynamic_pointer_cast<ImageTemplate<Type>>(m_image); }

		//	Converts to a device texture object for kernal access.
		operator devTexTemplate<Type>() const { return devTexTemplate<Type>(m_hTexture); }
	};

	/*********************************************************************
	**********************    TextureBase<float>    **********************
	*********************************************************************/

	//	Internal texture template for textures with floating-type texel format.
	template<template<typename> class ImageTemplate, template<typename> class devTexTemplate, typename Type> class TextureBase<ImageTemplate, devTexTemplate, Type, true> : public Texture
	{

	public:

		/**
		 *	@brief		Binds a texture memory as the texture resource.
		 *	@param[in]	image - Shared pointer to the memory object.
		 *	@param[in]	sampler - Sampler for texture fetched.
		 *	@throws		cudaError_t - In case of failure.
		 */
		void bind(std::shared_ptr<ImageTemplate<Type>> image, Sampler sampler = Sampler()) { this->bindImage(image, sampler, FormatMapping<Type>::value); }


		/**
		 *	@brief		Binds a texture memory as the texture resource.
		 *	@param[in]	image - Shared pointer to the memory object.
		 *	@param[in]	sampler - Sampler for texture fetched.
		 *	@details	Force set ReadMode::eNormalizedFloat to 1.
		 *	@throws		cudaError_t - In case of failure.
		 *	@todo		
		 */
		template<typename StorageType> void bind(std::shared_ptr<ImageTemplate<StorageType>> pImage, Sampler sampler = Sampler())
		{
			this->bindImage(pImage, sampler, FormatMapping<Type>::value);
		}

	public:

		//	Returns shared pointer to the binded texture memory.
		std::shared_ptr<ImageTemplate<void>> getImage() const { return std::dynamic_pointer_cast<ImageTemplate<void>>(m_image); }

		//	Converts to a device texture object for kernal access.
		operator devTexTemplate<Type>() const { return devTexTemplate<Type>(m_hTexture); }
	};
}
	/*********************************************************************
	***************************    Texture    ****************************
	*********************************************************************/

	template<typename Type> class Texture1D : public details::TextureBase<Image1D, dev::Tex1D, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture2D : public details::TextureBase<Image2D, dev::Tex2D, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture3D : public details::TextureBase<Image3D, dev::Tex3D, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class TextureCube : public details::TextureBase<ImageCube, dev::TexCube, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture1DLod : public details::TextureBase<Image1DLod, dev::Tex1DLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture2DLod : public details::TextureBase<Image2DLod, dev::Tex2DLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture3DLod : public details::TextureBase<Image3DLod, dev::Tex3DLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class TextureCubeLod : public details::TextureBase<ImageCubeLod, dev::TexCubeLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture1DLayered : public details::TextureBase<Image1DLayered, dev::Tex1DLayered, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture2DLayered : public details::TextureBase<Image2DLayered, dev::Tex2DLayered, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class TextureCubeLayered : public details::TextureBase<ImageCubeLayered, dev::TexCubeLayered, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture1DLayeredLod : public details::TextureBase<Image1DLayeredLod, dev::Tex1DLayeredLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class Texture2DLayeredLod : public details::TextureBase<Image2DLayeredLod, dev::Tex2DLayeredLod, Type, details::IsFloatingFormat<Type>::value> {};
	template<typename Type> class TextureCubeLayeredLod : public details::TextureBase<ImageCubeLayeredLod, dev::TexCubeLayeredLod, Type, details::IsFloatingFormat<Type>::value> {};
}