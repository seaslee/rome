#include "full_connected_layer.h"
#include "layer_factory.h"
#include "../matrix/random.h"

namespace snoopy {
namespace ml {

template<typename DataType>
void FCLayer<DataType>::init_spec_layer(const vector<Blob<DataType> *> & input_blob,
             const vector<Blob<DataType> *> & output_blob) {
    //check
    size_t input_blob_size = input_blob.size();
    size_t output_blob_size = output_blob.size();
    CHECK_EQ(input_blob_size, 1);
    CHECK_EQ(output_blob_size, 1);

    size_t input_blob_dim0 = input_blob[0]->dim_at(0);
    size_t input_blob_dim1 = input_blob[0]->dim_at(1);
    size_t output_blob_dim0 = output_blob[0]->dim_at(0);
    size_t output_blob_dim1 = output_blob[0]->dim_at(1);

    size_t in_nodes_dim = this->param_blob_[0]->dim_at(0);
    size_t out_nodes_dim = this->param_blob_[0]->dim_at(1);

    CHECK_EQ(input_blob_dim1, in_nodes_dim);
    CHECK_EQ(output_blob_dim1, out_nodes_dim);
    CHECK_NE(this->param_blob_[0]->get_data(), nullptr);
    CHECK_NE(this->param_blob_[0]->get_diff(), nullptr);

    n_in_ = input_blob_dim1;
    n_out_ = output_blob_dim1;
    n_nums_ = input_blob_dim0;
    is_add_bias_ = false;
    Matrix<DataType, 2> param_matrix = this->param_blob_[0]->get_data()->flatten_2d_matrix();

    //initialize the parameter
    float a = -1. / sqrt(n_in_);
    float b = -1. / sqrt(n_in_);
    Random::uniform(param_matrix, a, b); 
}

template<typename DataType>
void FCLayer<DataType>::reshape(const vector<Blob<DataType> *> & input_blob,
            const vector<Blob<DataType> *> & output_blob) {

}

template<typename DataType>
void FCLayer<DataType>::forward_cpu(const vector<Blob<DataType> *> & input_blob,
                 const vector<Blob<DataType> *> & output_blob) {
    Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> param_matrix = this->param_blob_[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
    out_matrix.copy_from(dot(input_matrix, param_matrix));
}

template<typename DataType>
void FCLayer<DataType>::backward_cpu(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
    //backward, transpose mat
    Matrix<DataType, 2>  output_diff_matrix = output_blob[0]->get_diff()->flatten_2d_matrix();
    Matrix<DataType, 2> param_matrix = this->param_blob_[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> param_diff_matrix = this->param_blob_[0]->get_diff()->flatten_2d_matrix();

    Matrix<DataType, 2>  in_diff_matrix = input_blob[0]->get_diff()->flatten_2d_matrix();
    Matrix<DataType, 2> in_data_maxtrix = input_blob[0]->get_data()->flatten_2d_matrix();
    //gradient with respect to input 
    size_t dim0 = this->param_blob_[0]->dim_at(0);
    size_t dim1 = this->param_blob_[0]->dim_at(1);
    MatrixShape<2> mat_shape(0, {dim1, dim0});
    Matrix<DataType, 2> trans_param_mat(mat_shape); //storage TODO @xinchao
    transpose(trans_param_mat, param_matrix);
    in_diff_matrix.copy_from(dot(output_diff_matrix, trans_param_mat));
    //gradient with respect to weights
    size_t in_dim0 = input_blob[0]->dim_at(0);
    size_t in_dim1 = input_blob[0]->dim_at(1);
    MatrixShape<2> trans_in_shape(0, {in_dim1, in_dim0});
    Matrix<DataType, 2> in_trans_mat(trans_in_shape);
    transpose(in_trans_mat, in_data_maxtrix);
    param_diff_matrix.copy_from(dot(in_trans_mat, output_diff_matrix));
}

//regesite
LAYER_REGISTER_CLASS(FC)

} //end namespace
} //end namespace


