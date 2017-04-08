#ifndef SNOOPY_IO_GET_CONF_H_
#define SNOOPY_IO_GET_CONF_H_

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "../proto/snoopy.pb.h"
#include <fcntl.h>
#include <cstdio>
#include "../common/com_def.h"
#include "../common/logging.h"
#include <fstream>
#include <iostream>
#include "../matrix/matrix_blob.h"

using std::string;
using std::shared_ptr;
using std::vector;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::ios;

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

template <typename DataType>
inline int load_model_from_binary_file(const string & file_name,
                      const vector<shared_ptr<matrix::Blob<DataType> > > & para_blobs) {
    size_t file_offset = 0;
    size_t blob_data_size = 0;
    ifstream file (file_name, ios::in|ios::binary);
    if (!file.is_open()) {
        LOG_FATAL << "open file : " << file_name << " failed!" << endl;
    }
    //file.seekg (0, ios::beg);

    for (int i = 0; i < para_blobs.size(); ++i) {
       blob_data_size = para_blobs[i]->get_count() * sizeof(DataType);
       if(!(file.read(para_blobs[i]->get_raw_data(), blob_data_size))) {
            LOG_FATAL << "read model failed!" << endl;         
       }
    }

    file.close();

    return snoopy::SUCCESS;
}

template <typename DataType>
inline int write_model_to_binary_file(const string & file_name,
                      const vector<shared_ptr<matrix::Blob<DataType> > > & para_blobs) {
    size_t file_offset = 0;
    size_t blob_data_size = 0;
    ofstream file (file_name, ios::out|ios::binary);
    if (!file.is_open()) {
        LOG_FATAL << "open file : " << file_name << " failed!" << endl;
    }
    for (int i = 0; i < para_blobs.size(); ++i) {
       blob_data_size = para_blobs[i]->get_count() * sizeof(DataType);
       std::cerr << "index: " << i << " blob_data_size: " << blob_data_size << endl;
       std::cerr << "dim0: " << para_blobs[i]->dim_at(0) << " dim1: " << 
           para_blobs[i]->dim_at(1) << endl;
      // write_matrix_to_binary_file(file, file_offset, para_blobs[i], blob_data_size); 
       file.write(static_cast<char *>(para_blobs[i]->get_raw_data()), blob_data_size);
       file_offset += blob_data_size;
    }
    file.close();

    return snoopy::SUCCESS;
}
                                  
}

}
#endif
