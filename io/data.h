#ifndef SNOOPY_IO_DATA_H
#define SNOOPY_IO_DATA_H

#include "../proto/snoopy.pb.h"
#include "../matrix/matrix_blob.h"

namespace snoopy {
namespace io{

template <typename DataType>
class DataFeed {
    public:
        virtual int get_data(std::vector<matrix::Blob<DataType> *> & output_blob) = 0;
};

}
}

#endif
