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
#pragma once

#include "macros.hpp"

#include <stdarg.h>
#include <functional>
#include <iostream>
#include <thread>
#include <string>

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Logger    ****************************
	*********************************************************************/

	/**
	 *	@brief		Lightweight logger.
	 */
	class Logger final
	{
		NS_NONCOPYABLE(Logger)

	private:

		//!	@brief		Default constructor.
		Logger() = default;

		//!	@brief		Default destructor.
		~Logger() = default;

	public:

		/**
		 *	@brief		Log levels.
		 */
		enum Level { eAssert, eError, eWarning, eInfo, eDebug };


		/**
		 *	@brief		Type of message callback function.
		 */
		using LogCallback = std::function<void(const char * fileName, int line, const char * funcName, Level eLevel, const char * logMsg)>;

	public:

		/**
		 *	@brief		Return the singleton instance.
		 *	@warning	Be cautious when multiple dynamic libraries call this function (no longer a singleton in that case).
		 */
		static Logger * getInstance()
		{
			static Logger s_instance;

			return &s_instance;
		}


		/**
		 *	@brief		Specify a callback function.
		 *	@param[in]	pfnCallback		User-defined callback function for receiving log message.
		 *	@note		Pass nullptr will unregister the callback function.
		 */
		void registerCallback(LogCallback pfnCallback) noexcept { m_pfnCallback = pfnCallback; }


		/**
		 *	@brief		Transfer log message to the specified callback function (if registered).
		 *	@param[in]	fileName	Which file sends this message.
		 *	@param[in]	line		Which line in the file invokes this function.
		 *	@param[in]	funcName	Which function invokes this functin.
		 *	@param[in]	eLevel		Log level of this message.
		 */
		void printLog(const char * fileName, int line, const char * funcName, Level eLevel, const char * format, ...)
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

	private:

		LogCallback		m_pfnCallback;
	};
}

/*************************************************************************
*************************    PrintLog_Macros    **************************
*************************************************************************/

#define NS_INFO_LOG(...)		NS_NAMESPACE::Logger::getInstance()->printLog(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eInfo, __VA_ARGS__)
#define NS_DEBUG_LOG(...)		NS_NAMESPACE::Logger::getInstance()->printLog(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eDebug, __VA_ARGS__)
#define NS_ERROR_LOG(...)		NS_NAMESPACE::Logger::getInstance()->printLog(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eError, __VA_ARGS__)
#define NS_WARNING_LOG(...)		NS_NAMESPACE::Logger::getInstance()->printLog(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eWarning, __VA_ARGS__)

#define NS_INFO_LOG_IF(condition, ...)			if (condition)	NS_INFO_LOG(__VA_ARGS__)
#define NS_ERROR_LOG_IF(condition, ...)			if (condition)	NS_ERROR_LOG(__VA_ARGS__)
#define NS_WARNING_LOG_IF(condition, ...)		if (condition)	NS_WARNING_LOG(__VA_ARGS__)
#define NS_ASSERT_LOG_IF(condition, ...)		if (condition)	NS_NAMESPACE::Logger::getInstance()->printLog(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eAssert, __VA_ARGS__);	NS_ASSERT(!(condition))

#if defined(DEBUG) || defined(_DEBUG)
	#define NS_DEBUG_LOG_IF(condition, ...)		if (condition)	NS_DEBUG_LOG(__VA_ARGS__)
#else
	#define NS_DEBUG_LOG_IF(condition, ...)
#endif