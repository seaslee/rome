#ifndef SNOOPY_MATRIX_NN_H
#define SNOOPY_MATRIX_NN_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include "layer.h"
#include "../common/utils.h"
#include "../matrix/matrix_blob.h"
#include "../proto/snoopy.pb.h"

//using namespace std;
using std::shared_ptr;
using std::vector;
using std::string;
namespace snoopy {
namespace ml {

template<typename DataType>
class NeuralNet {
public:
  explicit NeuralNet();
  explicit ~NeuralNet();
  /**
   * create net from net parameter 
   *
   */
  int init(const NetParameter & para);

  /**
   * perform the forward process to compute the output of each layer
   */
  const vector<Blob<DataType> *>& forward(DataType * loss);
  
  /**
   * compute the gradient w.r.t the parameter
   */
  void backprop();

  /**
   * update the parameter of net with computed gradient 
   */
  void update();

  private:
  string net_name_; //!< network name
  Phrase net_type_; //!< train or test

  /**
   * layers
   */
  vector<shared_ptr<Layer<DataType> > > layers_; //!< layers
  vector<string> layer_names_; //!< layer names
  unordered_map<string, int> layer_name_index_dict_; //!< layer index
  vector<bool> layer_need_bp_;

  /**
   * inter data for layers
   */
  vector<shared_ptr<Blob<DataType> > > blob_;
  vector<string> blob_names_; //! < blobs names
  unordered_map<string, int> blob_name_index_dict_; 
  vector<bool> blob_need_bp_;

  /**
   * recore the input blob for each layer
   */
  vector<vector<Blob<DataType> * > > bottom_blobs_;
  vector<int> bottom_blob_ids_;
  vector<bool> bottom_blob_need_bp_;

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

}
}

#endif //MATRIX_NN_H
