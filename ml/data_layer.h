#ifndef SNOOPY_ML_DATA_LAYER_H
#define SNOOPY_ML_DATA_LAYER_H

#include "layer.h"

namespace snoopy {
namespace ml{

template <typename DataType>
class DataFeedLayer : public Layer<DataType> {
    public:
       explicit DataFeedLayer(const LayerParameter & para) :
         Layer<DataType>(para) {}
        virtual void clear() = 0;
        virtual int read_file() = 0;
        virtual bool is_end() = 0;
        virtual int get_data(std::vector<matrix::Blob<DataType> *> & output_blob) = 0;

        virtual void reshape(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob) {}
    protected:
      virtual void forward_cpu(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob) {}

      virtual void backward_cpu(const vector<Blob<DataType> *> & input_blob,
                          const vector<bool> & need_bp,
                          const vector<Blob<DataType> *> & output_blob) {}
};

}
}

#endif
