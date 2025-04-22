add_executable(test_hash test_hash.cpp)
add_executable(test_offset test_offset.cpp)
add_executable(test_chunk test_chunk.cpp)
add_executable(test_filter test_filter.cpp)

target_link_libraries(test_hash PRIVATE GTest::gtest_main UNHASH)
target_link_libraries(test_offset PRIVATE GTest::gtest_main UNHASH)
target_link_libraries(test_chunk PRIVATE GTest::gtest_main UNHASH TBB::tbb)
target_link_libraries(test_filter PRIVATE GTest::gtest_main UNHASH TBB::tbb)

add_test(NAME TestHash COMMAND $<TARGET_FILE:test_hash>)
add_test(NAME TestOffset COMMAND $<TARGET_FILE:test_offset>)
add_test(NAME TestChunk COMMAND $<TARGET_FILE:test_chunk>)
add_test(NAME TestFilter COMMAND $<TARGET_FILE:test_filter>)