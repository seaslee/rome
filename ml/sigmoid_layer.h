#ifndef SNOOPY_ML_SIGMOID_LAYER_H_
#define SNOOPY_ML_SIGMOID_LAYER_H_

#include "inplace_layer.h"

namespace snoopy {
namespace ml {

template<typename DataType>
struct sigmoid {
  inline static DataType matrix_op(DataType x) {
    return static_cast<DataType>(1.0) / (1 + exp(-x));
  }
};

template <typename DataType>
class SigmoidLayer : public InplaceLayer<DataType> {
    public:
     explicit SigmoidLayer(const LayerParameter & para) :
         InplaceLayer<DataType>(para) {}
    protected:
      void act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s);
      void act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s);

      virtual void forward_cpu(const vector<Blob<DataType> *> & input_blob,
                        const vector<Blob<DataType> *> & output_blob);

      virtual void backward_cpu(const vector<Blob<DataType> *> & input_blob,
                          const vector<bool> & need_bp,
                          const vector<Blob<DataType> *> & output_blob); 
};

}
}


#endif
