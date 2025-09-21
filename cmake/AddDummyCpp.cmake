# AddDummyCpp.cmake
#
# Provides a function add_dummy_if_no_cpp(target)
# This will automatically add a dummy.cpp file to the given target
# if it has no .cpp sources. Mainly useful for Visual Studio so that
# the C/C++ property pages appear in the project settings.
#
# Usage:
#   add_dummy_if_no_cpp(Target)

function(add_dummy_if_no_cpp target)
	# Get the sources of the target
	get_target_property(sources ${target} SOURCES)

	if(NOT sources)
		set(sources "")
	endif()

	set(has_cpp FALSE)
	foreach(src ${sources})
		get_filename_component(ext ${src} EXT)
		if(ext STREQUAL ".cpp")
			set(has_cpp TRUE)
		endif()
	endforeach()

	# If no .cpp exists, create a dummy.cpp in the build dir
	if(NOT has_cpp)
		set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
		file(WRITE ${DUMMY_CPP}
			"//	Required to enable C/C++ property pages in Visual Studio.\n"
		)
		target_sources(${target} PRIVATE ${DUMMY_CPP})
	endif()
endfunction()
