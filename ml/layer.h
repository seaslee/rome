#ifndef SNOOPY_ML_LAYER_H_
#define SNOOPY_ML_LAYER_H_

//#include "../matrix/matrix.h"
#include "../common/utils.h"
#include "../proto/snoopy.pb.h"
#include "../matrix/matrix_blob.h"

using namespace snoopy::matrix;

namespace snoopy {
namespace ml {

template<typename DataType>
class Layer {
public:
     explicit Layer(const LayerParameter & para) :
         layer_param_(para), phrase_(para.phrase()) {
         if (layer_param_.blob_size() > 0) {
            param_blob_.resize(layer_param_.blob_size());            
            for (int i = 0; i < param_blob_.size(); ++i) {
                param_blob_[i] = create_blob_object<DataType>(layer_param_.blob(i), true);
            }
         }
     }

     virtual ~Layer() {}

     inline int init(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob) {
         init_spec_layer(input_blob, output_blob);
         reshape(input_blob, output_blob);
         //make sure 
         check_blob_count(input_blob, output_blob);
         return 0;
     }

     virtual void init_spec_layer(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob) = 0;

     virtual void reshape(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob) = 0;

     /**
     * compute the output blob of layer given the input blob of the layer
     *
     * @param input_blob:  the input data blob
     * @param output_blob: the output data blob
     *
     * @return the loss of the layer iff the nonzero loss weight
     */
     inline DataType forward(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob);

     /**
      * compute the gradient of input and learnable parameter(iff learnable parameter 
      * in layer)
      *
      * @param input_blob:  the input data blob
      * @param need_bp: indicate whether to progate the error the corresponding input blob 
      * @param output_blob: the output data blob
      *
      */
     inline void backward(const vector<Blob<DataType> *> & input_blob,
                          const vector<bool> & need_bp,
                          const vector<Blob<DataType> *> & output_blob);

     inline virtual int exact_bottom_blob() {return -1;}
     inline virtual int min_bottom_blob() {return -1;}
     inline virtual int max_bottom_blob() {return -1;}

     inline virtual int exact_top_blob() {return -1;}
     inline virtual int min_top_blob() {return -1;}
     inline virtual int max_top_blob() {return -1;}
     
     LayerParameter get_layer_parameter() {
        return layer_param_;
     }

     Phrase get_phrase() {
        return phrase_;
     }

     vector<shared_ptr<Blob<DataType> > > get_param_blob() {
        return param_blob_;
     }

 protected:
  LayerParameter layer_param_;
  Phrase phrase_;
  vector<shared_ptr<Blob<DataType> > > param_blob_;
  vector<bool> param_blob_need_bp_;

  virtual void forward_cpu(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob) = 0;

  virtual void backward_cpu(const vector<Blob<DataType> *> & input_blob,
                      const vector<bool> & need_bp,
                      const vector<Blob<DataType> *> & output_blob) = 0;

  virtual void check_blob_count(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob) {

    if (exact_bottom_blob() != -1) {
        CHECK_EQ(exact_bottom_blob(), input_blob.size());
    }

    if (min_bottom_blob() != -1) {
        CHECK_LE(min_bottom_blob(), input_blob.size());
    }

    if (max_bottom_blob() != -1) {
        CHECK_GE(max_bottom_blob(), input_blob.size());
    }

    if (exact_top_blob() != -1) {
        CHECK_EQ(exact_top_blob(), output_blob.size());
    }
    if (min_top_blob() != -1) {
        CHECK_LE(min_top_blob(), output_blob.size());
    }
    if (max_top_blob() != -1) {
        CHECK_GE(max_top_blob(), output_blob.size());
    }
  }

  DISALLOW_COPY_AND_ASSIGN(Layer);
};

template <typename DataType>
inline DataType Layer<DataType>::forward(const vector<Blob<DataType> *> & input_blob,
                const vector<Blob<DataType> *> & output_blob) {
    DataType loss(0);
    forward_cpu(input_blob, output_blob);
    return loss; //TODO
} 

template <typename DataType>
inline void Layer<DataType>::backward(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
    backward_cpu(input_blob, need_bp, output_blob);    
}

}
}

#endif /*ML_LAYER_H_*/
