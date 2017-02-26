MESSAGE(STATUS "Using bundled FindlibOpenBlas.cmake...")
FIND_PATH(
     OpenBlas_INCLUDE_DIR
     NAMES
     cblas.h 
     PATHS
     /usr/local/Cellar/openblas/0.2.18_2/include
)

FIND_LIBRARY(
   OpenBlas_LIBRARIES NAMES  openblas
   PATHS /usr/local/Cellar/openblas/0.2.18_2/lib
)
