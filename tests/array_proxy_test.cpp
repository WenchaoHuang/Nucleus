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

#include <nucleus/array_proxy.h>

/*********************************************************************************
*****************************    array_proxy_test    *****************************
*********************************************************************************/

void array_proxy_test()
{
	int a[3] = { 1, 2, 3 };
	std::array<int, 5> b = { 0,1,2,3,4 };
	std::vector<int> c = { 0,1 };

	ns::ArrayProxy<int> x0(a);
	ns::ArrayProxy<int> x1(b);
	ns::ArrayProxy<int> x2(c);
	ns::ArrayProxy<int> x3 = nullptr;
	ns::ArrayProxy<int> x4({ 1,2,3 });

	assert(x0[0] == a[0]);
	assert(x0[1] == a[1]);
	assert(x0[2] == a[2]);
	assert(x0.data() == a);
	assert(x0.size() == 3);
	assert(x0.empty() == false);

	assert(x1[0] == b[0]);
	assert(x1[1] == b[1]);
	assert(x1[2] == b[2]);
	assert(x1[3] == b[3]);
	assert(x1[4] == b[4]);
	assert(x1.data() == b.data());
	assert(x1.size() == b.size());
	assert(x1.empty() == false);

	assert(x2[0] == c[0]);
	assert(x2[1] == c[1]);
	assert(x2.data() == c.data());
	assert(x2.size() == c.size());
	assert(x2.empty() == false);

	assert(x3.size() == 0);
	assert(x3.data() == nullptr);
	assert(x3.empty() == true);

	if (!x4.empty())
	{
		auto x5 = x4.end();
		auto x6 = x4.data();
		auto x7 = x4.begin();
		auto x8 = x4.front();
		auto x9 = x4.back();
		assert(x4.size() == 3);
		assert(x4[0] == 1);
		assert(x4[1] == 2);
		assert(x4[2] == 3);
	}

	for (auto val : x4)
	{

	}
}