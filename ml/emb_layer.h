#ifndef SNOOPY_ML_EMB_LAYER_H_
#define SNOOPY_ML_EMB_LAYER_H_

#include "layer.h"
namespace snoopy {
namespace ml {

template<typename DataType>
class EmbeddingLayer : public Layer<DataType> {
public: 
     explicit EmbeddingLayer(const LayerParameter & para) :
         Layer<DataType>(para){
         if (para.has_emb_param()) {
            slot_capicity = para.emb_param().slot_capicity();
         }
         }
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

  int slot_capicity;

};
    
}
}

#endif
