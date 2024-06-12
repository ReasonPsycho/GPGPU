cmake_minimum_required(VERSION 3.28)
project(GPGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 26)
add_executable(GPGPU main.cu)

#Special thanks to UnaNancyOwen https://gist.github.com/UnaNancyOwen/5061d8c966178b753447e8a9f9ac8cf1
# Set OpenCVConfig.cmake Search Directory
set( OpenCV_DIR )
if( NOT CMAKE_CL_64 )
    set( OpenCV_DIR "C:/Tools/vcpkg/installed/x86-windows/share/opencv" )
else()
    set( OpenCV_DIR "C:/Tools/vcpkg/installed/x64-windows/share/opencv" )
endif()

# Find Package
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
    # Additional Include Directories
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    # Additional Library Directories
    link_directories( ${OpenCV_LIB_DIR} )

    # Additional Dependencies
    target_link_libraries( GPGPU ${OpenCV_LIBS} )
endif()

set_target_properties(GPGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
