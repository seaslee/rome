#ifndef SNOOPY_ML_INPLACE_LAYER_H_
#define SNOOPY_ML_INPLACE_LAYER_H_

#include "layer.h"
namespace snoopy {
namespace ml {

template <typename DataType>
class InplaceLayer : public Layer<DataType> {
    public:
        explicit InplaceLayer(const LayerParameter & para):
            Layer<DataType>(para) {}
         virtual void init_spec_layer(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob) {}

         virtual void reshape(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob) {}

         virtual int exact_bottom_blob() { return 1; }
         virtual int exact_top_blob() { return 1; }
};

}
}
#endif
