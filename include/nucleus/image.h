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
#include <vector>

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Image    *****************************
	*********************************************************************/

	/**
	 *	@brief		Base class represents a arbitrary texture memory.
	 *	@note		Texture memory are opaque memory layouts optimized for texture fetching.
	 */
	class Image
	{
		
	protected:

		/**
		 *	@brief		Constructs a image.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit Image(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, int flags);


		/**
		 *	@brief		Constructs from ImageLod.
		 *	@param[in]	texHandle - Handle of texture memory (from cudaMipmappedArray_t).
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 */
		explicit Image(cudaArray_t texHandle, Format format, size_t width, size_t height, size_t depth, int flags);


		/**
		 *	@brief		Virtual destructor.
		 */
		virtual ~Image() noexcept;

	public:

		//	Retruns the width of the image.
		uint32_t width() const { return m_width; }

		//	Returns the texel format of the image.
		Format format() const { return m_format; }

		//	Returns CUDA type of this object.
		cudaArray_t getHandle() const { return m_texHandle; }

		//	Returns pointer to the allocator associated with.
		std::shared_ptr<DeviceAllocator> getAllocator() const { return m_allocator; }

		//	Checks if the buffer supports surface load/store operations.
		bool isSurfaceLoadStoreSupported() const;

	protected:
        
        const cudaArray_t							m_texHandle;
		const std::shared_ptr<DeviceAllocator>		m_allocator;
		const Format								m_format;
        const uint32_t								m_width;
        const uint32_t								m_height;
		const uint32_t								m_depth;
		const int									m_flags;
	};

	/*********************************************************************
	***************************    ImageLod    ***************************
	*********************************************************************/

	/**
	 *	@brief		Base class represents a arbitrary mipmapped texture memory.
	 *  @note		Texture memory are opaque memory layouts optimized for texture fetching.
	 *	@see		class Image
	 */
	class ImageLod
	{

	protected:

		/**
		 *	@brief		Constructs a image with level of details.
		 *	@param[in]	allocator - Pointer to the associated allocator.
		 *	@param[in]	format - Texel format of the image.
		 *	@param[in]	width - Width of the image.
		 *	@param[in]	height - height of the image.
		 *	@param[in]	depth - Depth of the image.
		 *	@param[in]	numLevels - Number of mipmap levels to allocated.
		 *	@param[in]	flags - Flags for image creation (interanl use).
		 *	@throw		cudaError_t - In case of failure.
		 */
		explicit ImageLod(std::shared_ptr<DeviceAllocator> allocator, Format format, size_t width, size_t height, size_t depth, unsigned int numLevels, int flags);


		/**
		 *	@brief		Virtual destructor.
		 */
		virtual ~ImageLod() noexcept;

	public:

		//	Retruns the width of the image.
		uint32_t width() const { return m_width; }

		//	Returns the texel format of the image.
		Format format() const { return m_format; }

		//	Returns the number of mipmap levels.
		unsigned int numLevels() const { return m_numLevels; }

		//	Returns CUDA type of this object.
		cudaMipmappedArray_t getHandle() const { return m_texHandle; }

		//	Returns pointer to the allocator associated with.
		std::shared_ptr<DeviceAllocator> getAllocator() const { return m_allocator; }

	protected:

		//	Returns CUDA type of mipmap objects.
		std::vector<cudaArray_t> getMipmapHandles() const;

	protected:

		const cudaMipmappedArray_t					m_texHandle;
		const std::shared_ptr<DeviceAllocator>		m_allocator;
		const unsigned int							m_numLevels;
		const Format								m_format;
		const uint32_t								m_width;
		const uint32_t								m_height;
		const uint32_t								m_depth;
		const int									m_flags;
	};
}