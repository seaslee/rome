#include "emb_layer.h"
#include "layer_factory.h"

namespace snoopy {
namespace ml {
/*
 * Blob: M * N, M is sample num and N is sample 
 *
 */
template<typename DataType>
void EmbeddingLayer<DataType>::init_spec_layer(const vector<Blob<DataType> *> & input_blob,
             const vector<Blob<DataType> *> & output_blob) {
    //check
    size_t input_blob_size = input_blob.size();
    size_t output_blob_size = output_blob.size();
    CHECK_EQ(input_blob_size, 1);
    CHECK_EQ(output_blob_size, 1);
    //check input_blob, output_blob shape
    size_t input_dim0 = input_blob[0]->dim_at(0);
    size_t input_dim1 = input_blob[0]->dim_at(1);
    CHECK_EQ(input_dim1, slot_capicity);

    size_t output_dim0 = output_blob[0]->dim_at(0);
    size_t output_dim1 = output_blob[0]->dim_at(1);
    CHECK_EQ(output_dim0, input_dim0 * slot_capicity);
    CHECK_EQ(output_dim1, this->param_blob_[0]->dim_at(1));
}

template<typename DataType>
void EmbeddingLayer<DataType>::reshape(const vector<Blob<DataType> *> & input_blob,
            const vector<Blob<DataType> *> & output_blob) {

}

template<typename DataType>
void EmbeddingLayer<DataType>::forward_cpu(const vector<Blob<DataType> *> & input_blob,
                 const vector<Blob<DataType> *> & output_blob) {
    Matrix<DataType, 2> input_matrix = input_blob[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> out_matrix = output_blob[0]->get_data()->flatten_2d_matrix();
    Matrix<DataType, 2> param_matrix = this->param_blob_[0]->get_data()->flatten_2d_matrix();

    for (size_t i = 0; i < input_matrix.get_row(); ++i) {
        for (size_t j = 0; j < input_matrix.get_column(); ++j) {
            int index = input_matrix[i][j];
            if (index < 0) {
                out_matrix[i*slot_capicity+j].clear_data();
                continue;
            } else if(index > param_matrix.get_row()) {
               LOG_FATAL << "index :" << index << " should be less than " << param_matrix.get_row();
            }
            out_matrix[i*slot_capicity+j].copy_from(param_matrix[index]);
        }
    }
}

template<typename DataType>
void EmbeddingLayer<DataType>::backward_cpu(const vector<Blob<DataType> *> & input_blob,
                  const vector<bool> & need_bp,
                  const vector<Blob<DataType> *> & output_blob) {
    //dot nothing
}

//regesite
LAYER_REGISTER_CLASS(Embedding)

} //end namespace
} //end namespace


