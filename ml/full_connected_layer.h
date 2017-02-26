#ifndef SNOOPY_ML_FULL_CONNECTED_LAYER_H_
#define SNOOPY_ML_FULL_CONNECTED_LAYER_H_

#include "layer.h"
namespace snoopy {
namespace ml {

template<typename DataType>
class FCLayer : public Layer<DataType> {
public: 
     explicit FCLayer(const LayerParameter & para) :
         Layer<DataType>(para) {}
     virtual void init_spec_layer(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob);

     virtual void reshape(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob);

     virtual int exact_bottom_blob() { return 1; }
     virtual int exact_top_blob() { return 1; }

protected:
  virtual void forward_cpu(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob);

  virtual void backward_cpu(const vector<Blob<DataType> *> & input_blob,
                      const vector<bool> & need_bp,
                      const vector<Blob<DataType> *> & output_blob);

  size_t n_in_;
  size_t n_out_;
  size_t n_nums_;
  int is_add_bias_; //TODO
};
    
}
}

#endif
