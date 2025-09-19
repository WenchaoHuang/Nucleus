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

#include "logger.h"

#include <stdarg.h>
#include <iostream>
#include <thread>
#include <string>

NS_USING_NAMESPACE

/*********************************************************************************
**********************************    Logger    **********************************
*********************************************************************************/

void Logger::log(const char * fileName, int line, const char * funcName, Level level, const char * format, ...)
{
	va_list argPtr;

	va_start(argPtr, format);

	thread_local std::string logString;

	logString.resize(static_cast<size_t>(_vscprintf(format, argPtr)) + 1);

	vsprintf_s(const_cast<char*>(logString.data()), logString.size(), format, argPtr);

	va_end(argPtr);

	//	ANSI color codes
	constexpr const char * COLOR_RED = "\033[31m";
	constexpr const char * COLOR_RESET = "\033[0m";
	constexpr const char * COLOR_GREEN = "\033[32m";
	constexpr const char * COLOR_MAGENTA = "\033[35m";
	constexpr const char * COLOR_ORANGE = "\033[38;2;255;165;0m";
#ifdef NS_DEBUG
	constexpr const char * COLOR_CYAN = "\033[36m";
#endif

	if (m_pfnCallback != nullptr)
	{
		m_pfnCallback(fileName, line, funcName, level, logString.c_str());
	}
	else if (level == Level::Assert)
	{
		std::cout << COLOR_MAGENTA << " [" << std::this_thread::get_id() << "] Assert: " << funcName << "() => " << logString << COLOR_RESET << std::endl;
	}
	else if (level == Level::Error)
	{
		std::cout << COLOR_RED << " [" << std::this_thread::get_id() << "] Error: " << funcName << "() => " << logString << COLOR_RESET << std::endl;
	}
	else if (level == Level::Warning)
	{
		std::cout << COLOR_ORANGE << " [" << std::this_thread::get_id() << "] Warning: " << funcName << "() => " << logString << COLOR_RESET << std::endl;
	}
	else if (level == Level::Info)
	{
		std::cout << COLOR_GREEN << " [" << std::this_thread::get_id() << "] Info: " << funcName << "() => " << logString << COLOR_RESET << std::endl;
	}
	else if (level == Level::Debug)
	{
	#ifdef NS_DEBUG
		std::cout << COLOR_CYAN << " [" << std::this_thread::get_id() << "] Debug: " << funcName << "() => " << logString << COLOR_RESET << std::endl;
	#endif
	}
}