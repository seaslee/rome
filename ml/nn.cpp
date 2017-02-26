#include "nn.h"

namespace snoopy {
namespace ml {

template <typename DataType>
explicit NeuralNet::NeuralNet() {

}

template <typename DataType>
explicit NeuralNet::~NeuralNet() {

}

template <typename DataType>
int NeuralNet::init(const NetParameter & para) {
    net_name_ = para.name();
    net_type_ = para.state().netphrase();

    //layers
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++layer_index) {
        LayerParameter lp = para.layer_param(layer_index);
        layers_.push_back(LayerRegiste::create_layer(lp));
        layer_names_.push_back(lp.name());
        layer_name_index_dict_[lp.name()] = layer_index;
        layer_need_bp_.push_back(lp.is_bp());
    }

    //input data
    //TODO: add input class for network
    shared_ptr<Blob<DataType> > input_blob = create_blob_object<DataType>(
            para.layer_para(0).b_blob_shape());
    blob_.push_back(input_blob);
    blob_name_index_dict_[para.layer_para(0).b_blob_name()] = 0;

    //layer inter data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++layer_index) {
        shared_ptr<Blob<DataType> > tmp_blob = create_blob_object<DataType>(
                para.layer_para(layer_index).t_blob_shape());
        blob_.push_back(tmp_blob);
        blob_name_index_dict_[para.layer_para(0).t_blob_name()] = layer_index + 1;
    }

    for (int layer_index = 0; layer_index < para.layer_param_size(); ++ layer_index) {
        vector<Blob<DataType> * > > layer_bottom_blobs_;
        vector<Blob<DataType> * > > layer_top_blobs_;

        for (int b_blob_index = 0; b_blob_index < para.layer_para(
                    layer_index).b_blob_name_size(); ++b_blob_index) {
            layer_bottom_blobs_.push_back(blob_[blob_name_index_dict_[
                    para.layer_para(layer_index_).b_blob_name(i)]]);
        }

        for (int t_blob_index = 0; t_blob_index < para.layer_para(
                    layer_index).t_blob_name_size(); ++b_blob_index) {
            layer_bottom_blobs_.push_back(blob_[blob_name_index_dict_[
                    para.layer_para(layer_index_).b_blob_name(i)]]);
        }
    }

    //layer param data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++ layer_index) {
        for (int para_blob_index = 0; para_blob_index < layers_[layer_index].get_param_blob.size(); ++para_blob_index) {
           para_blobs_.push_back(layers_[layer_index].get_param_blob[i]);
        }
    }
}

} //end namespace
} //end namespace