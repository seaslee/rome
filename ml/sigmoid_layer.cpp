#include "sigmoid_layer.h"
#include "layer_factory.h"

namespace snoopy {
namespace ml {

template<typename DataType>
void SigmoidLayer<DataType>::act_fun(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = single_op<sigmoid<DataType>>(s);
}

template<typename DataType>
void SigmoidLayer<DataType>::act_fun_de(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s) {
    t = s * (1 - s);
}

template<typename DataType>
void SigmoidLayer<DataType>::forward_cpu(const vector<Blob<DataType> *> & input_blob,
                const vector<Blob<DataType> *> & output_blob) {
  //check 
  Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
  Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
  act_fun(out_matrix, input_matrix);
}

template<typename DataType>
void SigmoidLayer<DataType>::backward_cpu(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
  Matrix<DataType, 2>  in_diff_matrix = input_blob[0]->get_diff()->flatten_2d_matrix();
  Matrix<DataType, 2>  output_diff_matrix = output_blob[0]->get_diff()->flatten_2d_matrix();
  Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
  act_fun_de(in_diff_matrix, out_matrix);
  in_diff_matrix = in_diff_matrix * output_diff_matrix;
}


//regesite
LAYER_REGISTER_CLASS(Sigmoid)

}
}
