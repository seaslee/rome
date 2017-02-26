project(test)
cmake_minimum_required(VERSION 3.6)
set(CMAKE_VERBOSE_MAKEFILE ON)
 
find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS snoopy.proto)
add_library(snoopy_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(snoopy_proto ${PROTOBUF_LIBRARIES} pthread)
