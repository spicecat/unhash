include_directories("${CMAKE_SOURCE_DIR}/include")

add_executable(unhash_temp temp.cpp)
add_sycl_to_target(TARGET unhash_temp SOURCES temp.cpp)
target_compile_options(unhash_temp PRIVATE -O3 -march=native -DNDEBUG -fsycl-device-code-split=per_kernel -tbb)
target_link_libraries(unhash_temp PRIVATE UNHASH)

add_executable(unhash main.cpp)
add_sycl_to_target(TARGET unhash SOURCES main.cpp)
target_compile_options(unhash PRIVATE -O3 -march=native -DNDEBUG -fsycl-device-code-split=per_kernel -tbb)
target_link_libraries(unhash PRIVATE UNHASH)

add_executable(unhash_bench bench.cpp)
add_sycl_to_target(TARGET unhash_bench SOURCES bench.cpp)
target_compile_options(unhash_bench PRIVATE -O3 -march=native -DNDEBUG -fsycl-device-code-split=per_kernel)
target_link_libraries(unhash_bench PRIVATE UNHASH)

add_executable(unhash_chunk chunk.cpp)
add_sycl_to_target(TARGET unhash_chunk SOURCES chunk.cpp)
target_compile_options(unhash_chunk PRIVATE -O3 -march=native -DNDEBUG -fsycl-device-code-split=per_kernel)
target_link_libraries(unhash_chunk PRIVATE UNHASH)

add_custom_target(run-temp COMMAND ./unhash_temp DEPENDS unhash_temp)
add_custom_target(run COMMAND ./unhash DEPENDS unhash)
add_custom_target(run-bench COMMAND ./unhash_bench DEPENDS unhash_bench)
add_custom_target(run-chunk COMMAND ./unhash_chunk DEPENDS unhash_chunk)