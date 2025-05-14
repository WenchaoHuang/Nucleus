/**
 *	Copyright (c) 2025 Huang Wenchao <physhuangwenchao@gmail.com>
 *
 *	All rights reserved. Use of this source code is governed by a
 *	GPL-2.0 license that can be found in the LICENSE file.
 *
 *	This program is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *	GNU General Public License for more details.
 */

#include "logger.hpp"

#include <stdarg.h>
#include <iostream>
#include <thread>
#include <string>

NS_USING_NAMESPACE

/*************************************************************************
******************************    Logger    ******************************
*************************************************************************/

void Logger::log(const char * fileName, int line, const char * funcName, Level eLevel, const char * format, ...)
{
	va_list argPtr;

	va_start(argPtr, format);

	thread_local std::string logString;

	logString.resize(static_cast<size_t>(_vscprintf(format, argPtr)) + 1);

	vsprintf_s(const_cast<char*>(logString.data()), logString.size(), format, argPtr);

	va_end(argPtr);

	if (m_pfnCallback != nullptr)
	{
		m_pfnCallback(fileName, line, funcName, eLevel, logString.c_str());
	}
	else if (eLevel == Level::eAssert)
	{
		std::cout << " [" << std::this_thread::get_id() << "] Assert: " << funcName << "() => " << logString << std::endl;
	}
	else if (eLevel == Level::eError)
	{
		std::cout << " [" << std::this_thread::get_id() << "] Error: " << funcName << "() => " << logString << std::endl;
	}
	else if (eLevel == Level::eWarning)
	{
		std::cout << " [" << std::this_thread::get_id() << "] Warning: " << funcName << "() => " << logString << std::endl;
	}
	else if (eLevel == Level::eInfo)
	{
		std::cout << " [" << std::this_thread::get_id() << "] Info: " << funcName << "() => " << logString << std::endl;
	}
	else if (eLevel == Level::eDebug)
	{
	#if defined(DEBUG) || defined(_DEBUG)
		std::cout << " [" << std::this_thread::get_id() << "] Debug: " << funcName << "() => " << logString << std::endl;
	#endif
	}
}