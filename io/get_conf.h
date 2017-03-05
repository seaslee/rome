#ifndef SNOOPY_IO_GET_CONF_H_
#define SNOOPY_IO_GET_CONF_H_

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "../proto/snoopy.pb.h"

namespace snoopy {
namespace io {

int read_net_proto_from_text_file(const string & f_name, 
                                NetParamter & net_param);
                                  
}
}
#endif
