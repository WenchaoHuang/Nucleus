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
#include <functional>

namespace NS_NAMESPACE
{
	/*********************************************************************
	****************************    Logger    ****************************
	*********************************************************************/

	/**
	 *	@brief		Lightweight logger (singleton).
	 */
	class Logger
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
		 *	@param[in]	pfnCallback - User-defined callback function for receiving log message.
		 *	@note		Pass nullptr will unregister the callback function.
		 */
		void registerCallback(LogCallback pfnCallback) noexcept { m_pfnCallback = pfnCallback; }


		/**
		 *	@brief		Transfer log message to the specified callback function (if registered).
		 *	@param[in]	fileName - Which file sends this message.
		 *	@param[in]	line - Which line in the file invokes this function.
		 *	@param[in]	funcName - Which function invokes this functin.
		 *	@param[in]	eLevel - Log level of this message.
		 */
		void log(const char * fileName, int line, const char * funcName, Level eLevel, const char * format, ...);

	private:

		LogCallback		m_pfnCallback;
	};
}

/*************************************************************************
****************************    Log Macros    ****************************
*************************************************************************/

#define NS_INFO_LOG(...)		NS_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eInfo, __VA_ARGS__)
#define NS_DEBUG_LOG(...)		NS_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eDebug, __VA_ARGS__)
#define NS_ERROR_LOG(...)		NS_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eError, __VA_ARGS__)
#define NS_WARNING_LOG(...)		NS_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eWarning, __VA_ARGS__)

#define NS_INFO_LOG_IF(condition, ...)			if (condition)	NS_INFO_LOG(__VA_ARGS__)
#define NS_ERROR_LOG_IF(condition, ...)			if (condition)	NS_ERROR_LOG(__VA_ARGS__)
#define NS_WARNING_LOG_IF(condition, ...)		if (condition)	NS_WARNING_LOG(__VA_ARGS__)
#define NS_ASSERT_LOG_IF(condition, ...)		if (condition)	NS_NAMESPACE::Logger::getInstance()->log(__FILE__, __LINE__, __FUNCTION__, NS_NAMESPACE::Logger::eAssert, __VA_ARGS__);	NS_ASSERT(!(condition))

#if defined(DEBUG) || defined(_DEBUG)
	#define NS_DEBUG_LOG_IF(condition, ...)		if (condition)	NS_DEBUG_LOG(__VA_ARGS__)
#else
	#define NS_DEBUG_LOG_IF(condition, ...)
#endif