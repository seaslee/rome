#ifndef SNOOPY_ML_SOFTSIGN_LAYER_H_
#define SNOOPY_ML_SOFTSIGN_LAYER_H_

#include "inplace_layer.h"
#include <cfloat>

namespace snoopy {
namespace ml {

template<typename DataType>
struct softsign {
  inline static DataType matrix_op(DataType x) {
#ifdef USE_DOUBLE
    if (x < DBL_MAX)
    return x > -DBL_MAX ? x / (1 + abs(x)) : -1;
    return 1;
#else
    if (x < FLT_MAX)
      return x > -FLT_MAX ? x / (1 + abs(x)) : -1;
    return 1;
#endif
  }
};

template<typename DataType>
struct fabs_map {
  inline static DataType matrix_op(DataType x) {
    return fabs(x);
  }
};

template<typename DataType>
struct square_map {
  inline static DataType matrix_op(DataType x) {
    return x * x;
  }
};


template <typename DataType>
class SoftsignLayer : public InplaceLayer<DataType> {
    public:
     explicit SoftsignLayer(const LayerParameter & para) :
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
