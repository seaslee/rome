MESSAGE(STATUS "Using bundled FindlibProtobuf.cmake...")
FIND_PATH(
     PROTOBUF_INCLUDE_DIRS
     PATHS
     /usr/local/include/google/protobuf
)

FIND_LIBRARY(
    PROTOBUF_LIBRARIES
    NAMES  protobuf
   PATHS /usr/local/lib
)
