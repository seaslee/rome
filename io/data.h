

#include "../proto/snoopy.pb.h"

namespace snoopy {
namespace io{

class DataFeed {
    public:
        virtual int get_data(vector<Blob<DataType> *> & output_blob) = 0;
    private:
        DataFeedParameter data_param_;
};


}
}
