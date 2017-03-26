#include "softmax_with_loss_layer.h"
#include "layer_factory.h"

namespace snoopy {
namespace ml {

template<typename DataType>
void SoftmaxWithLossLayer<DataType>::init_spec_layer(const vector<Blob<DataType> *> & input_blob,
             const vector<Blob<DataType> *> & output_blob) {
    //check
    size_t input_blob_size = input_blob.size();
    size_t output_blob_size = output_blob.size();
    CHECK_EQ(input_blob_size, 2);
    CHECK_EQ(output_blob_size, 1);
    CHECK_EQ(input_blob[0]->get_blobshape(), output_blob[0]->get_blobshape());
}

template<typename DataType>
void SoftmaxWithLossLayer<DataType>::reshape(const vector<Blob<DataType> *> & input_blob,
            const vector<Blob<DataType> *> & output_blob) {

}

template<typename DataType>
void SoftmaxWithLossLayer<DataType>::forward_cpu(const vector<Blob<DataType> *> & input_blob,
                 const vector<Blob<DataType> *> & output_blob) {
    Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
    softmax(out_matrix, input_matrix);
}

template<typename DataType>
void SoftmaxWithLossLayer<DataType>::backward_cpu(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
    //backward, compute the gradient on the input directly
    //see ref: http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
    Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
    //Matrix<DataType, 2> label_matrix = input_blob[1]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();

    Matrix<DataType, 2>  in_diff_matrix = input_blob[0]->get_diff()->flatten_2d_matrix();

    size_t row_n = input_matrix.get_row();
    size_t col_n = input_matrix.get_column();

    Matrix<DataType, 2> label_matrix = input_blob[1]->get_data()->flatten_2d_matrix();
    size_t label_row_n = label_matrix.get_row();
    size_t label_col_n = label_matrix.get_column(); 
    CHECK_EQ(row_n, label_row_n);
    CHECK_GE(label_col_n, 1);

    for (size_t index_sample = 0; index_sample < row_n; ++index_sample) {
        //int label = static_cast<int>(input_blob[1]->get_data_at(index_sample)) - 1;
        int label = static_cast<int>(label_matrix[index_sample][0]);
        for (size_t index_dim = 0; index_dim < col_n; ++index_dim) {
            if (label == index_dim) {
                in_diff_matrix[index_sample][index_dim] = out_matrix[index_sample][index_dim] - 1;            
            } else {
                in_diff_matrix[index_sample][index_dim] = out_matrix[index_sample][index_dim];            
            }
        }
    }
}

//regesite
LAYER_REGISTER_CLASS(SoftmaxWithLoss)

} //end namespace
} //end namespace


