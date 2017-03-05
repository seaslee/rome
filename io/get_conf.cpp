#include "get_conf.h"
#include <fcntl.h>
#include <cstdio>
#include "../common/com_def.h"
#include "../common/logging.h"

namespace snoopy {
namespace io {

int read_net_proto_from_text_file(const string & f_name, 
                                NetParamter & net_param) {

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
                                  
}
}
