#ifndef SNOOPY_ML_RELU_LAYER_H_
#define SNOOPY_ML_RELU_LAYER_H_

#include "inplace_layer.h"
#include "layer_factory.h"
#include <cfloat>

namespace snoopy {
namespace ml {

template<typename DataType>
struct rectfier {
  inline static DataType matrix_op(DataType x) {
    return std::max(static_cast<DataType>(0), x);
  }
};

template<typename DataType>
struct rectfier_deri_map {
  inline static DataType matrix_op(DataType x) {
    return x > 0 ? 1 : 0;
  }
};

template <typename DataType>
class RELULayer : public InplaceLayer<DataType> {
    public:
     explicit RELULayer(const LayerParameter & para) :
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
