# project 
project(snoopy)
cmake_minimum_required(VERSION 3.6) 
option(test "Build all tests." OFF)

set(CMAKE_SOURCE_DIR .)
set(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules ${CMAKE_SOURCE_DIR}/cmake/modules) 

#c++11 support
set(CMAKE_CXX_STANDARD 11)

# 查找当前目录下的源文件
aux_source_directory(. DIR_SRCS)

# 添加子目录
add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(matrix)
add_subdirectory(storage)
add_subdirectory(ml)


# 指定生成目标
add_executable(main  main.cpp)

##openblas
find_package(libOpenBlas REQUIRED)
mark_as_advanced(
    OpenBlas_INCLUDE_DIR
    OpenBlas_LIBRARIES 
)
if (OpenBlas_INCLUDE_DIR AND OpenBlas_LIBRARIES)
  include_directories(${OpenBlas_INCLUDE_DIR})
  target_link_libraries(main ${OpenBlas_LIBRARIES})
endif (OpenBlas_INCLUDE_DIR AND OpenBlas_LIBRARIES)


################
#test
################
if (test) 
add_executable(matrix_test  test/matrix_test.cc)
add_executable(layer_test  test/layer_test.cc)
include_directories(${OpenBlas_INCLUDE_DIR})
target_link_libraries(matrix_test ${OpenBlas_LIBRARIES} libgtest )
add_dependencies(ml snoopy_proto )
add_dependencies(layer_test snoopy_proto )
target_link_libraries(layer_test ${OpenBlas_LIBRARIES} libgtest ml snoopy_proto)

# We need thread support
find_package(Threads REQUIRED)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/master.zip
    URL_HASH MD5=d87f3b2bd89c7df3841fb2c9e0b51b6e
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest )

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)

# Set libgmock properties
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")

endif()
