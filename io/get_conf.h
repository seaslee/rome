#ifndef SNOOPY_IO_GET_CONF_H_
#define SNOOPY_IO_GET_CONF_H_

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "../proto/snoopy.pb.h"
#include <fcntl.h>
#include <cstdio>
#include "../common/com_def.h"
#include "../common/logging.h"


namespace snoopy {
namespace io {

inline int read_net_proto_from_text_file(const string & f_name, 
                                NetParameter & net_param) {
    int file_handle = open(f_name.c_str(), O_RDONLY);
    if (!file_handle) {
        LOG_FATAL << "net configure file: " << f_name <<" not exists!";
        return snoopy::FAILURE;
    }
    google::protobuf::io::FileInputStream file_input(file_handle);
    if (!google::protobuf::TextFormat::Parse(&file_input, &net_param)) {
        LOG_FATAL << "net configure file: " << f_name <<" read failed!";
        return snoopy::FAILURE;
    }
    return snoopy::SUCCESS;
}


inline int read_solve_proto_from_text_file(const string & f_name, 
                                SolverParameter & solver_param) {
    int file_handle = open(f_name.c_str(), O_RDONLY);
    if (!file_handle) {
        LOG_FATAL << "solver configure file: " << f_name <<" not exists!";
        return snoopy::FAILURE;
    }
    google::protobuf::io::FileInputStream file_input(file_handle);
    if (!google::protobuf::TextFormat::Parse(&file_input, &solver_param)) {
        LOG_FATAL << "net configure file: " << f_name <<" read failed!";
        return snoopy::FAILURE;
    }
    return snoopy::SUCCESS;
}
                                  
}
}
#endif
