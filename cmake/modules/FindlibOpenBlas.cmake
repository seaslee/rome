MESSAGE(STATUS "Using bundled FindlibOpenBlas.cmake...")
FIND_PATH(
     OpenBlas_INCLUDE_DIR
     NAMES
     cblas.h 
     PATHS
     /usr/local/OpenBLAS/include
)

FIND_LIBRARY(
   OpenBlas_LIBRARIES NAMES  openblas
   PATHS /usr/local/OpenBLAS/lib
)
