cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

list(APPEND CMAKE_PREFIX_PATH "/home/libtorch/")
find_package(Torch REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
target_link_libraries(example "${TORCH_LIBRARIES}")
target_link_libraries(anyExecutable Boost::program_options)
set_property(TARGET example PROPERTY CXX_STANDARD 14)

## The following code block is suggested to be used on Windows.
## According to https://github.com/pytorch/pytorch/issues/25457,
## the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET example-app
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:example-app>)
#endif (MSVC)
