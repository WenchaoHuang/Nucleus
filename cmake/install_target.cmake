# Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Include necessary CMake modules for installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Define a function to install a target and its associated files.
function(install_target target_name)
	
	# Derive package metadata from the project that calls this function.
	set(package_name "${PROJECT_NAME}")
	
	# Resolve package metadata relative to the project directory that calls the function.
	set(install_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${package_name}")
	set(config_file "${CMAKE_CURRENT_BINARY_DIR}/${package_name}Config.cmake")
	set(version_file "${CMAKE_CURRENT_BINARY_DIR}/${package_name}ConfigVersion.cmake")
	set(package_config_template "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${package_name}Config.cmake.in")
	
	# Values consumed by @PACKAGE_NAME@ and @PACKAGE_TARGETS_FILE@ in the
	# package config template passed to configure_package_config_file().
	set(PACKAGE_NAME "${package_name}")
	set(PACKAGE_TARGETS_FILE "${package_name}Targets.cmake")
	
	# Check if the target exists.
	if(NOT TARGET ${target_name})
		message(FATAL_ERROR "Cannot install unknown target: ${target_name}")
	endif()
	
	# Check if the package config template exists.
	if(NOT EXISTS "${package_config_template}")
		message(FATAL_ERROR "Package config template not found: ${package_config_template}")
	endif()
	
	# Install the target and its associated files.
	install(TARGETS ${target_name}
		EXPORT ${package_name}Targets
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	)
	
	# Install the public headers.
	install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${target_name}
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		FILES_MATCHING
		PATTERN "*.cuh"
		PATTERN "*.hpp"
		PATTERN "*.h"
	)
	
	# Install generated version and export headers.
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/${target_name}_export.h
		${CMAKE_CURRENT_BINARY_DIR}/${target_name}_version.h
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${target_name}
	)
	
	# Install linker debug symbols for Debug builds on MSVC.
	if(MSVC)
		get_target_property(target_type ${target_name} TYPE)
		if(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY" OR target_type STREQUAL "EXECUTABLE")
			install(FILES $<TARGET_PDB_FILE:${target_name}>
				DESTINATION ${CMAKE_INSTALL_BINDIR}
				CONFIGURATIONS Debug
				OPTIONAL
			)
		elseif(target_type STREQUAL "STATIC_LIBRARY")
			install(FILES $<TARGET_FILE_DIR:${target_name}>/$<TARGET_FILE_BASE_NAME:${target_name}>.pdb
				DESTINATION ${CMAKE_INSTALL_LIBDIR}
				CONFIGURATIONS Debug
				OPTIONAL
			)
		endif()
	endif()
	
	# Create and install CMake package configuration files.
	configure_package_config_file(
		${package_config_template}
		${config_file}
		INSTALL_DESTINATION ${install_cmake_dir}
	)
	
	# Create and install CMake package version file.
	write_basic_package_version_file(
		${version_file}
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY SameMajorVersion
	)
	
	# Install the export targets and configuration files.
	install(EXPORT ${package_name}Targets
		FILE ${package_name}Targets.cmake
		NAMESPACE ${package_name}::
		DESTINATION ${install_cmake_dir}
	)
	
	# Install the package configuration and version files.
	install(FILES
		${config_file}
		${version_file}
		DESTINATION ${install_cmake_dir}
	)
endfunction()
