#include "vsum_layer.h"
#include "layer_factory.h"

namespace snoopy {
namespace ml {

template<typename DataType>
void VsumLayer<DataType>::forward_cpu(const vector<Blob<DataType> *> & input_blob,
                 const vector<Blob<DataType> *> & output_blob) {
  Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
  Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
  size_t in_row = input_matrix.get_row();
  size_t out_row = out_matrix.get_row();
  size_t sum_range = in_row / out_row;
  for (size_t i = 0; i < out_row; ++i) {
    Matrix<DataType, 2> slice_matrix1 = out_matrix.slice(i, i+1); 
    Matrix<DataType, 2> slice_matrix2 = input_matrix.slice(i*sum_range, (i+1)*sum_range);
    sum(slice_matrix1, slice_matrix2, 0);
  }
}

template<typename DataType>
void VsumLayer<DataType>::backward_cpu(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
    Matrix<DataType, 2>  output_diff_matrix = output_blob[0]->get_diff()->flatten_2d_matrix();
    Matrix<DataType, 2>  in_diff_matrix = input_blob[0]->get_diff()->flatten_2d_matrix();

    size_t in_row = in_diff_matrix.get_row();
    size_t out_row = output_diff_matrix.get_row();
    size_t sum_range = in_row / out_row;
    for (size_t i = 0; i < out_row; ++i) {
        Matrix<DataType, 2> slice_matrix = in_diff_matrix.slice(i*sum_range, (i+1)*sum_range); 
        repmat(slice_matrix, output_diff_matrix.slice(i, i+1), 0);
    }
}
//regesite
LAYER_REGISTER_CLASS(Vsum)

} //end namespace
} //end namespace


