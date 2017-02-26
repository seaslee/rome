# 查找当前目录下的源文件
AUX_SOURCE_DIRECTORY(. DIR_ML_SRCS)

include_directories(${OpenBlas_INCLUDE_DIR})
# 添加链接库
ADD_LIBRARY(ml ${DIR_ML_SRCS})
