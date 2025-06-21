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

#include <nucleus/stream.h>
#include <nucleus/device.h>
#include <nucleus/context.h>
#include <nucleus/texture.h>
#include <nucleus/image_1d.h>
#include <nucleus/image_2d.h>
#include <nucleus/image_3d.h>
#include <nucleus/image_cube.h>
#include <nucleus/launch_utils.cuh>

/*************************************************************************
***************************    test_texture    ***************************
*************************************************************************/

__global__ void test_device_texture(dev::Tex1D<int> tex0, dev::Tex1DLod<int> tex1, dev::Tex1DLayered<int> tex2, dev::Tex1DLayeredLod<int> tex3,
									dev::Tex2D<float> tex4, dev::Tex2DLod<int> tex5, dev::Tex2DLayered<int> tex6, dev::Tex2DLayeredLod<int> tex7,
									dev::Tex3D<int> tex8, dev::Tex3DLod<int> tex9,
									dev::TexCube<int> tex10, dev::TexCubeLod<int> tex11, dev::TexCubeLayered<int> tex12, dev::TexCubeLayeredLod<int> tex13)
{
	//	1D
	tex0.empty();
	tex0.fetch(0.5f);
	tex0.getHandle();

	tex1.empty();
	tex1.fetch(0.5f, 0);
	tex1.Grad(0.0f, 1.0f, 2.0f);
	tex1.getHandle();

	tex2.empty();
	tex2.fetch(0.5f, 0);
	tex2.getHandle();
	
	tex3.empty();
	tex3.fetch(0.5f, 0, 0.5f);
	tex3.Grad(0.0f, 1, 1.0f, 2.0f);
	tex3.getHandle();

	//	2D
	tex4.empty();
	tex4.fetch(0.5f, 0.5f);
	tex4.getHandle();

	tex5.empty();
	tex5.fetch(0.5f, 0.2f, 0);
	tex5.Grad(0.0f, 1.0f, { 2.0f, 1.0f }, { 1.0f, 2.0f });
	tex5.getHandle();

	tex6.empty();
	tex6.fetch(0.5f, 0.0f, 0);
	tex6.getHandle();

	tex7.empty();
	tex7.fetch(0.5f, 1.0f, 0, 0.5f);
	tex7.Grad(0.0f, 2.0f, 1, { 1.0f, 1.0f }, { 2.0f, 2.0f });
	tex7.getHandle();

	//	3D
	tex8.empty();
	tex8.fetch(0.5f, 0.5f, 2.0f);
	tex8.getHandle();

	tex9.empty();
	tex9.fetch(0.5f, 0.2f, 1.0f, 0);
	tex9.Grad(0.0f, 1.0f, 2.0f, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex9.getHandle();

	//	Cube
	tex10.empty();
	tex10.fetch(0.5f, 0.2f, 0.3f);
	tex10.getHandle();

	tex11.empty();
	tex11.fetch(0.5f, 0.6f, 0.8f, 0);
	tex11.Grad(0.0f, 1.0f, 2.0f, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex11.getHandle();

	tex12.empty();
	tex12.fetch(0.5f, 2.0f, 3.0f, 0);
	tex12.getHandle();

	tex13.empty();
	tex13.fetch(0.5f, 0.3f, 0.1f, 0, 0.5f);
	tex13.Grad(0.5f, 0.3f, 0.1f, 1, { 2.0f, 1.0f, 2.0f, 2.0f }, { 1.0f, 2.0f, 2.0f, 3.0f });
	tex13.getHandle();
}


void test_texture()
{
	auto device = ns::Context::getInstance()->getDevice(0);
	auto allocator = device->getDefaultAllocator();
	auto & stream = device->getDefaultStream();

	//	1D
	ns::Texture1D<int>	texture0;
	texture0.bind(std::make_shared<ns::Image1D<int>>(allocator, 10));
	assert(texture0.getImage() != nullptr);
	
	ns::Texture1DLod<int>	texture1;
	texture1.bind(std::make_shared<ns::Image1DLod<int>>(allocator, 10, 2));
	assert(texture1.getImage() != nullptr);

	ns::Texture1DLayered<int>	texture2;
	texture2.bind(std::make_shared<ns::Image1DLayered<int>>(allocator, 10, 2));
	assert(texture2.getImage() != nullptr);

	ns::Texture1DLayeredLod<int>	texture3;
	texture3.bind(std::make_shared<ns::Image1DLayeredLod<int>>(allocator, 10, 2, 2));
	assert(texture3.getImage() != nullptr);

	//	2D
	ns::Texture2D<float>	texture4;
	texture4.bind(std::make_shared<ns::Image2D<char>>(allocator, 10, 10));
	texture4.bind(std::make_shared<ns::Image2D<float>>(allocator, 10, 10));
	assert(texture4.getImage() != nullptr);

	ns::Texture2DLod<int>	texture5;
	texture5.bind(std::make_shared<ns::Image2DLod<int>>(allocator, 10, 10, 2));
	assert(texture5.getImage() != nullptr);

	ns::Texture2DLayered<int>	texture6;
	texture6.bind(std::make_shared<ns::Image2DLayered<int>>(allocator, 10, 10, 2));
	assert(texture6.getImage() != nullptr);

	ns::Texture2DLayeredLod<int>	texture7;
	texture7.bind(std::make_shared<ns::Image2DLayeredLod<int>>(allocator, 10, 10, 2, 2));
	assert(texture7.getImage() != nullptr);

	//	3D
	ns::Texture3D<int>	texture8;
	texture8.bind(std::make_shared<ns::Image3D<int>>(allocator, 10, 10, 10));
	assert(texture8.getImage() != nullptr);

	ns::Texture3DLod<int>	texture9;
	texture9.bind(std::make_shared<ns::Image3DLod<int>>(allocator, 10, 10, 10, 2));
	assert(texture9.getImage() != nullptr);

	//	Cube
	ns::TextureCube<int>	texture10;
	texture10.bind(std::make_shared<ns::ImageCube<int>>(allocator, 10));
	assert(texture10.getImage() != nullptr);

	ns::TextureCubeLod<int>	texture11;
	texture11.bind(std::make_shared<ns::ImageCubeLod<int>>(allocator, 10, 2));
	assert(texture11.getImage() != nullptr);

	ns::TextureCubeLayered<int>	texture12;
	texture12.bind(std::make_shared<ns::ImageCubeLayered<int>>(allocator, 10, 2));
	assert(texture12.getImage() != nullptr);

	ns::TextureCubeLayeredLod<int>	texture13;
	texture13.bind(std::make_shared<ns::ImageCubeLayeredLod<int>>(allocator, 10, 2, 2));
	assert(texture13.getImage() != nullptr);

	stream.launch(test_device_texture, 1, 1)(texture0, texture1, texture2, texture3,
											 texture4, texture5, texture6, texture7,
											 texture8, texture9,
											 texture10, texture11, texture12, texture13);
	stream.sync();
}