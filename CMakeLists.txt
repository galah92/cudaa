cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.0.0
)
FetchContent_MakeAvailable(fmt)

# Disable response files for CUDA since clangd doesn't seem to support them
# https://github.com/clangd/vscode-clangd/issues/592#issuecomment-2036788981
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDAToolkit REQUIRED)

add_executable(cudaa src/main.cu)
target_link_libraries(cudaa PRIVATE fmt::fmt)
