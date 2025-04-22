cmake_minimum_required(VERSION 3.23.5)
set(CMAKE_CXX_COMPILER "icpx")

project(unhash
    VERSION 0.1
    DESCRIPTION "Unhash TheLogFather project"
    HOMEPAGE_URL "https://github.com/spicecat/unhash"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IntelSYCL REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(GTest REQUIRED)

add_library(UNHASH INTERFACE)
target_include_directories(UNHASH INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(UNHASH INTERFACE TBB::tbb)

add_subdirectory(src)

include(CTest)
enable_testing()
add_subdirectory(test)