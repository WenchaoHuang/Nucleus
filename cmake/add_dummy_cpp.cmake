# add_dummy_cpp.cmake
#
# Provides a function add_dummy_cpp(target)
# This will automatically add a dummy.cpp file to the given target.
#
# Usage:
#	add_dummy_cpp(Target)

function(add_dummy_cpp target)
	set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
	file(WRITE ${DUMMY_CPP} "//	Required to enable C/C++ property page in Visual Studio when no *.cpp files exist.\n" )
	target_sources(${target} PRIVATE ${DUMMY_CPP})
endfunction()