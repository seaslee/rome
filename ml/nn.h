#ifndef SNOOPY_ML_NN_H_
#define SNOOPY_ML_NN_H_

#include <vector>
#include <map>
#include "layer.h"
#include "../common/com_def.h"
#include "layer_factory.h"

//using namespace std;
using std::shared_ptr;
using std::vector;
using std::string;

namespace snoopy {
namespace ml {

template<typename DataType>
class NeuralNet {
public:
  NeuralNet() {}
  ~NeuralNet() {}
  /**
   * create net from net parameter 
   *
   */
  int init(const NetParameter & para);

  /**
   * perform the forward process to compute the output of each layer
   */
  void forward(DataType * loss);
  
  /**
   * compute the gradient w.r.t the parameter
   */
  void backprop();

  /**
   * update the parameter of net with computed gradient 
   */
  void update();

  private:
  int _i;
  string net_name_; //!< network name
  Phrase net_type_; //!< train or test

  /**
   * layers
   */
  vector<shared_ptr<Layer<DataType> > > layers_; //!< layers
  vector<string> layer_names_; //!< layer names
  map<string, int> layer_name_index_dict_; //!< layer index
  vector<bool> layer_need_bp_;

  /**
   * inter data for layers
   */
  vector<shared_ptr<Blob<DataType> > > blob_;
  vector<string> blob_names_; //! < blobs names
  map<string, int> blob_name_index_dict_; 
  vector<bool> blob_need_bp_;

  /**
   * recore the input blob for each layer
   */
  vector<vector<Blob<DataType> * > > bottom_blobs_;
  vector<vector<int> > bottom_blob_ids_;
  vector<vector<bool> > bottom_blob_need_bp_;

  /**
   * recore the output blob for each layer
   */
  vector<vector<Blob<DataType> * > > top_blobs_;
  vector<vector<int> > top_blob_ids_;
  vector<vector<bool> >top_blob_need_bp_;
  
  /**
   * input and output blob
   */
  vector<int> input_blob_ids_;
  vector<Blob<DataType> * > input_blob_ptrs_;
  vector<int> output_blob_ids_;
  vector<Blob<DataType> * > output_blob_ptrs_;

  /**
   * parameters
   */
  vector<shared_ptr<Blob<DataType> > > para_blobs_;
  vector<Blob<DataType> * > learnable_para_blobs_;

  vector<int> learnable_para_ids_;
  vector<float> learnable_para_lr_;
  vector<bool> has_learnable_para_lr_;
};

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
    int net_blob_index(0);
    for (int in_blob_index = 0; in_blob_index < para.input_param().slot_size(); ++ in_blob_index) {
        BlobShape input_blob_shape {static_cast<unsigned long>(para.input_param().batch_size()), 
                                    static_cast<unsigned long>(para.input_param().slot_capicity())};
        shared_ptr<Blob<DataType> > input_blob = create_blob_object<DataType>(
                input_blob_shape, false);
        blob_.push_back(input_blob);
        blob_name_index_dict_[para.input_param().t_blob_name()[in_blob_index]] = net_blob_index++;
    }

    //layer output data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++layer_index) {
        for (int t_blob_index = 0; t_blob_index < para.layer_param(layer_index).t_blob_name_size(); 
                ++t_blob_index) {
            shared_ptr<Blob<DataType> > tmp_blob = create_blob_object<DataType>(
                    para.layer_param(layer_index).t_blob_shape(t_blob_index), true);
            blob_.push_back(tmp_blob);
            blob_name_index_dict_[para.layer_param(layer_index).t_blob_name(t_blob_index)] = net_blob_index++;
        }
    }

    for (auto iter = blob_name_index_dict_.begin(); iter != blob_name_index_dict_.end(); ++iter) {
        cerr << iter->first << "\t" << blob_[iter->second]->dim_at(0) << "\t" << blob_[iter->second]->dim_at(1) << endl;
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
            layer_bottom_blob_ids.push_back(index_of_net_blob);
        }

        for (int t_blob_index = 0; t_blob_index < para.layer_param(
                    layer_index).t_blob_name_size(); ++t_blob_index) {
            size_t index_of_net_blob = blob_name_index_dict_[
                    para.layer_param(layer_index).t_blob_name(t_blob_index)];
            layer_top_blobs.push_back(blob_[index_of_net_blob].get());
            layer_top_blob_ids.push_back(index_of_net_blob);
        }
        bottom_blob_ids_.push_back(layer_bottom_blob_ids);
        top_blob_ids_.push_back(layer_top_blob_ids);

        bottom_blobs_.push_back(layer_bottom_blobs);
        top_blobs_.push_back(layer_top_blobs);
    }

    //layer param data
    for (int layer_index = 0; layer_index < para.layer_param_size(); ++ layer_index) {
        for (int para_blob_index = 0; para_blob_index < layers_[layer_index]->get_param_blob().size(); 
                ++para_blob_index) {
           para_blobs_.push_back(layers_[layer_index]->get_param_blob()[para_blob_index]);
        }
    }
    return snoopy::SUCCESS;
}

template <typename DataType>
void NeuralNet<DataType>::forward(DataType * loss) {
    *loss = 0;
    for (int layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        shared_ptr<Layer<DataType> > layer = layers_[layer_index];
        *loss += layer->forward(bottom_blobs_[layer_index], top_blobs_[layer_index]);
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
    int i = 0;
    return;
}


}
}

#endif //MATRIX_NN_H

