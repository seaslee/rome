#include "nn.h"
#include <vector>
#include "layer_factory.h"
#include "../common/com_def.h"

namespace snoopy {
namespace ml {

template <typename DataType>
int NeuralNet<DataType>::init(const NetParameter & para) {
    if (para.has_name()) {
        net_name_ = para.name();
    } else {
        LOG_FATAL << "Check net configure, init failed for missing net name!";
        return snoopy::FAILURE;
    }
    if (para.has_state()) {
        net_type_ = para.state().netphrase();
    } else {
        LOG_FATAL << "Check net configure, init failed for missing net status!";
        return snoopy::FAILURE;
    }

    //layers
    if (para.layer_param_size() == 0) {
        LOG_FATAL << "Check the net configure, init a network without layers!";
        return snoopy::FAILURE;
    }

    for (int layer_index = 0; layer_index < para.layer_param_size(); ++layer_index) {
        LayerParameter lp = para.layer_param(layer_index);
        layers_.push_back(LayerRegiste<DataType>::create_layer(lp));
        layer_names_.push_back(lp.name());
        layer_name_index_dict_[lp.name()] = layer_index;
        layer_need_bp_.push_back(lp.is_bp());
    }

    //input data
    size_t net_blob_index(0);
    for (int in_blob_index = 0; in_blob_index < para.input_param().slot_size(); ++ in_blob_index) {
        BlobShape input_blob_shape {static_cast<unsigned long>(para.input_param().batch_size()), static_cast<unsigned long>(para.input_param().slot_capicity())};
        shared_ptr<Blob<DataType> > input_blob = create_blob_object<DataType>(
                input_blob_shape);
        blob_.push_back(input_blob);
        blob_name_index_dict_[para.input_param().t_blob_name()] = net_blob_index;
    }

    //layer output data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++layer_index) {
        shared_ptr<Blob<DataType> > tmp_blob = create_blob_object<DataType>(
                para.layer_param(layer_index).t_blob_shape());
        blob_.push_back(tmp_blob);
        blob_name_index_dict_[para.layer_param(0).t_blob_name()] = net_blob_index;
    }

    //allocate the input and output for layers
    std::vector<Blob<DataType> * > layer_bottom_blobs;
    std::vector<int> layer_bottom_blob_ids;
    std::vector<bool> layer_bottom_blob_need_bp;
    std::vector<Blob<DataType> * > layer_top_blobs;
    std::vector<int> layer_top_blob_ids;
    std::vector<bool> layer_top_blob_need_bp;

    for (int layer_index = 0; layer_index < para.layer_param_size(); ++ layer_index) {
        layer_bottom_blobs.clear();
        layer_bottom_blob_ids.clear();
        layer_bottom_blob_need_bp.clear();
        layer_top_blobs.clear();
        layer_top_blob_ids.clear();
        layer_top_blob_need_bp.clear();

        for (int b_blob_index = 0; b_blob_index < para.layer_param(
                    layer_index).b_blob_name_size(); ++b_blob_index) {
            size_t index_of_net_blob = blob_name_index_dict_[
                    para.layer_param(layer_index).b_blob_name(b_blob_index)];
            layer_bottom_blobs.push_back(blob_[index_of_net_blob].get());
            bottom_blob_ids_.push_back(index_of_net_blob);
        }

        for (int t_blob_index = 0; t_blob_index < para.layer_param(
                    layer_index).t_blob_name_size(); ++t_blob_index) {
            size_t index_of_net_blob = blob_name_index_dict_[
                    para.layer_param(layer_index).t_blob_name(t_blob_index)];
            layer_top_blobs.push_back(blob_[index_of_net_blob].get());
            top_blob_ids_.push_back(index_of_net_blob);
        }

        bottom_blobs_.push_back(layer_bottom_blobs);
        top_blobs_.push_back(layer_top_blobs);
    }

    //layer param data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++ layer_index) {
        for (int para_blob_index = 0; para_blob_index < layers_[layer_index].get_param_blob.size(); ++para_blob_index) {
           para_blobs_.push_back(layers_[layer_index].get_param_blob[para_blob_index]);
        }
    }
    return snoopy::SUCCESS;
}

template <typename DataType>
void NeuralNet<DataType>::forward(DataType * loss) {
    *loss = 0;
    for (int layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        shared_ptr<Layer<DataType> > layer = layers_[layer_index];
        loss += layer->forward(bottom_blobs_[layer_index], top_blobs_[layer_index]);
    }
}

template <typename DataType>
void NeuralNet<DataType>::backprop() {
    for (int layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        vector<bool> need_bp;
        shared_ptr<Layer<DataType> > layer = layers_[layer_index];
        layer->backward(bottom_blobs_[layer_index], need_bp, top_blobs_[layer_index]);
    }
    
}

template <typename DataType>
void NeuralNet<DataType>::update() {

}

} //end namespace
} //end namespace
