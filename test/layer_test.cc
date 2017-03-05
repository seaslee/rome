#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h> 
#include "../proto/snoopy.pb.h"
#include "../ml/full_connected_layer.h"
#include "../ml/sigmoid_layer.h"
#include "../ml/emb_layer.h"
#include "../ml/vsum_layer.h"
#include "../ml/softmax_with_loss_layer.h"
#include "../ml/nn.h"

using namespace snoopy::ml;
using namespace snoopy;

int main(int argc, char** argv) { 
    testing::InitGoogleTest(&argc, argv); 
    // Runs all tests using Google Test. 
    return RUN_ALL_TESTS(); 
}

TEST(FCLayer, forward_backward) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 2> m2 { { 2, 3 }, { 2, 3 }, {2, 3} };

  Matrix<float, 2> m3 = dot(m1, m2, 1.f, false);
  Matrix<float, 2> m4 = dot(m1, m2);
  Matrix<float, 2> m5 {{12, 18}, {18, 27}};

  LayerParameter lp;
  lp.set_name("fc1");
  lp.set_type("FC");
  lp.set_bottom("fc1_bottom");
  lp.set_top("fc1_top");
  lp.set_phrase(TRAIN);

  BlobParameter * blob_param = lp.add_blob();
  BlobShapeProto *bsp = new BlobShapeProto; 
  bsp->add_dim(4);
  bsp->add_dim(5);
  blob_param->set_allocated_shape(bsp);

  LearnRateParameter * lr = new LearnRateParameter;
  lr->set_lr_multi(1.0);
  lr->set_lr_delay(1.0);
  lp.set_allocated_lr(lr);

  lp.set_loss_weight(1.0);
  lp.set_is_bp(false);

  FCLayerParameter * fc = new FCLayerParameter;
  fc->set_in_nodes_dim(4);
  fc->set_out_nodes_dim(5);
  lp.set_allocated_fc_param(fc);

  Layer<float> * fc_layer = new FCLayer<float>(lp);
  BlobShape in_blob_shape {3, 4};
  BlobShape out_blob_shape {3, 5};
  shared_ptr<Blob<float> > in_blob = create_blob_object<float>(in_blob_shape, true);
  shared_ptr<Blob<float> > out_blob = create_blob_object<float>(out_blob_shape, true);
  Matrix<float, 2>  in_data_matrix = in_blob->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_data_matrix = {{1, 2, 3, 4}, 
                                         {1, 3, 4, 5}, 
                                         {1, 2, 2, 3}};
  in_data_matrix.copy_from(tmp_in_data_matrix);
  Matrix<float, 2>  in_diff_matrix = in_blob->get_diff()->flatten_2d_matrix();
  Matrix<float, 2>  out_data_matrix = out_blob->get_data()->flatten_2d_matrix();
  Matrix<float, 2>  out_diff_matrix = out_blob->get_diff()->flatten_2d_matrix();
  Matrix<float, 2> tmp_out_diff_matrix = {{1, 2, 3, 6, 1}, 
                                          {1, 3, 5, 7, 4}, 
                                          {1, 2, 2, 1, 1}};
  out_diff_matrix.copy_from(tmp_out_diff_matrix);


  vector<Blob<float> *>  input_blob_vec;
  vector<Blob<float> *>  output_blob_vec;
  vector<bool> need_bp;
  input_blob_vec.push_back(in_blob.get());
  output_blob_vec.push_back(out_blob.get());
  need_bp.push_back(true);

  fc_layer->init(input_blob_vec, output_blob_vec);

  Matrix<float, 2> para_matrix = fc_layer->get_param_blob()[0]->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_learn_param = {{1, 2, 3, 6, 3}, 
                                      {1, 3, 5, 7, 4}, 
                                      {1, 2, 2, 1, 5},
                                      {1, 2, 2, 2, 6}};
  para_matrix.copy_from(tmp_learn_param);
  fc_layer->forward(input_blob_vec, output_blob_vec);

  Matrix<float, 2> exp_out {{10, 22, 27, 31, 50},
                            {13, 29, 36, 41, 65},
                            {8, 18, 23, 28, 39}};
  Matrix<float, 2> new_out_mat = out_blob->get_data()->flatten_2d_matrix();
  EXPECT_EQ(exp_out, new_out_mat);

  fc_layer->backward(input_blob_vec, need_bp, output_blob_vec);
  Matrix<float, 2> exp_diff {{53,  68,  22,  29}, 
                             {76, 100,  44,  55}, 
                             {20,  28,  15,  17}};
  Matrix<float, 2> new_in_diff = in_blob->get_diff()->flatten_2d_matrix();
  EXPECT_EQ(new_in_diff, exp_diff);
}


TEST(SigmoidLayer, forward_backward) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 2> m2 { { 2, 3 }, { 2, 3 }, {2, 3} };

  Matrix<float, 2> m3 = dot(m1, m2, 1.f, false);
  Matrix<float, 2> m4 = dot(m1, m2);
  Matrix<float, 2> m5 {{12, 18}, {18, 27}};

  LayerParameter lp;
  lp.set_name("sigmoid1");
  lp.set_type("Sigmoid");
  lp.set_bottom("sigmoid1_bottom");
  lp.set_top("sigmoid1_top");
  lp.set_phrase(TRAIN);

  Layer<float> * sigmoid_layer = new SigmoidLayer<float>(lp);
  BlobShape in_blob_shape {4, 3};
  BlobShape out_blob_shape {4, 3};

  shared_ptr<Blob<float> > in_blob = create_blob_object<float>(in_blob_shape, true);
  shared_ptr<Blob<float> > out_blob = create_blob_object<float>(out_blob_shape, true);
  Matrix<float, 2>  in_data_matrix = in_blob->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_data_matrix = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  in_data_matrix.copy_from(tmp_in_data_matrix);
  Matrix<float, 2>  in_diff_matrix = in_blob->get_diff()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_diff_matrix = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  in_diff_matrix.copy_from(tmp_in_diff_matrix);

  Matrix<float, 2>  out_data_matrix = out_blob->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_out_data_matrix = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  out_data_matrix.copy_from(tmp_out_data_matrix);
  Matrix<float, 2>  out_diff_matrix = out_blob->get_diff()->flatten_2d_matrix();
  Matrix<float, 2> tmp_out_diff_matrix = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  out_diff_matrix.copy_from(tmp_out_diff_matrix);

  vector<Blob<float> *>  input_blob_vec;
  vector<Blob<float> *>  output_blob_vec;
  vector<bool> need_bp;
  input_blob_vec.push_back(in_blob.get());
  output_blob_vec.push_back(out_blob.get());
  need_bp.push_back(true);

  sigmoid_layer->init(input_blob_vec, output_blob_vec);
  sigmoid_layer->forward(input_blob_vec, output_blob_vec);

  Matrix<float, 2> exp_out {{0.73105858,  0.88079708,  0.95257413}, 
                            {0.73105858,  0.95257413,  0.99330715},
                            {0.73105858,  0.88079708,  0.88079708}, 
                            {0.88079708,  0.95257413,  0.95257413}};
  Matrix<float, 2> new_out_mat = out_blob->get_data()->flatten_2d_matrix();
  EXPECT_EQ(exp_out, new_out_mat);

  sigmoid_layer->backward(input_blob_vec, need_bp, output_blob_vec);
  Matrix<float, 2> exp_diff {{0.19661193,  0.20998717,  0.13552998},
                             {0.19661193,  0.13552998,  0.03324028},
                             {0.19661193,  0.20998717,  0.20998717},
                             {0.20998717,  0.13552998,  0.13552998}};
  Matrix<float, 2> new_in_diff = in_blob->get_diff()->flatten_2d_matrix();
  EXPECT_EQ(new_in_diff, exp_diff);
}

TEST(EmbeddingLayer, forward_backward) {
  LayerParameter lp;
  lp.set_name("emb1");
  lp.set_type("Embedding");
  lp.set_bottom("emb1_bottom");
  lp.set_top("emb1_top");
  lp.set_phrase(TRAIN);

  BlobParameter * blob_param = lp.add_blob();
  BlobShapeProto *bsp = new BlobShapeProto; 
  bsp->add_dim(5);
  bsp->add_dim(3);
  blob_param->set_allocated_shape(bsp);

  LearnRateParameter * lr = new LearnRateParameter;
  lr->set_lr_multi(1.0);
  lr->set_lr_delay(1.0);
  lp.set_allocated_lr(lr);

  lp.set_loss_weight(1.0);
  lp.set_is_bp(false);

  EmbeddingParameter * emb = new EmbeddingParameter;
  emb->set_slot_capicity(5);
  lp.set_allocated_emb_param(emb);

  Layer<float> * emb_layer = new EmbeddingLayer<float>(lp);
  BlobShape in_blob_shape {4, 5};
  BlobShape out_blob_shape {20, 3};
  shared_ptr<Blob<float> > in_blob = create_blob_object<float>(in_blob_shape, true);
  shared_ptr<Blob<float> > out_blob = create_blob_object<float>(out_blob_shape, true);
  Matrix<float, 2>  in_data_matrix = in_blob->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_data_matrix = {{1, 2, -1, -1, -1}, 
                                         {2, 3, 3, -1, -1}, 
                                         {1, 0, 0, 0, 0},
                                         {4, 4, 4, 4, 0}};
  in_data_matrix.copy_from(tmp_in_data_matrix);

  vector<Blob<float> *>  input_blob_vec;
  vector<Blob<float> *>  output_blob_vec;
  vector<bool> need_bp;
  input_blob_vec.push_back(in_blob.get());
  output_blob_vec.push_back(out_blob.get());
  need_bp.push_back(true);

  emb_layer->init(input_blob_vec, output_blob_vec);

  Matrix<float, 2> para_matrix = emb_layer->get_param_blob()[0]->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_learn_param = {{1, 1, 1}, 
                                      {2, 2, 2}, 
                                      {3, 3, 3},
                                      {4, 4, 4},
                                      {5, 5, 5}};
  para_matrix.copy_from(tmp_learn_param);
  emb_layer->forward(input_blob_vec, output_blob_vec);

  Matrix<float, 2> exp_out {{2, 2, 2},
                            {3, 3, 3},
                            {0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0},
                            {3, 3, 3},
                            {4, 4, 4},
                            {4, 4, 4},
                            {0, 0, 0},
                            {0, 0, 0},
                            {2, 2, 2},
                            {1, 1, 1},
                            {1, 1, 1},
                            {1, 1, 1},
                            {1, 1, 1},
                            {5, 5, 5},
                            {5, 5, 5},
                            {5, 5, 5},
                            {5, 5, 5},
                            {1, 1, 1}
                            };
  Matrix<float, 2> new_out_mat = out_blob->get_data()->flatten_2d_matrix();
  EXPECT_EQ(exp_out, new_out_mat);

  //vsum test
  LayerParameter lp1;
  lp1.set_name("vsum1");
  lp1.set_type("Vsum");
  lp1.set_bottom("vsum1_bottom");
  lp1.set_top("vsum1_top");
  lp1.set_phrase(TRAIN);

  LearnRateParameter * lr1 = new LearnRateParameter;
  lr1->set_lr_multi(1.0);
  lr1->set_lr_delay(1.0);
  lp1.set_allocated_lr(lr1);
  lp1.set_loss_weight(1.0);
  lp1.set_is_bp(false);

  Layer<float> * vsum_layer = new VsumLayer<float>(lp1);
  BlobShape in_blob_shape1 {20, 3};
  BlobShape out_blob_shape1 {4, 3};
  shared_ptr<Blob<float> > in_blob1 = create_blob_object<float>(in_blob_shape1, true);
  shared_ptr<Blob<float> > out_blob1 = create_blob_object<float>(out_blob_shape1, true);
  Matrix<float, 2>  in_data_matrix1 = in_blob1->get_data()->flatten_2d_matrix();
  in_data_matrix1.copy_from(exp_out);
  Matrix<float, 2>  out_diff_matrix1 = out_blob1->get_diff()->flatten_2d_matrix();
  Matrix<float, 2> tmp_out_diff_matrix1 = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  out_diff_matrix1.copy_from(tmp_out_diff_matrix1);

  vector<Blob<float> *>  input_blob_vec1;
  vector<Blob<float> *>  output_blob_vec1;
  vector<bool> need_bp1;
  input_blob_vec1.push_back(in_blob1.get());
  output_blob_vec1.push_back(out_blob1.get());
  need_bp1.push_back(true);

  vsum_layer->init(input_blob_vec1, output_blob_vec1);

  Matrix<float, 2> exp_out1 {{5, 5, 5},
                            {11, 11, 11},
                            {6, 6, 6},
                            {21, 21, 21}};
  vsum_layer->forward(input_blob_vec1, output_blob_vec1);
  Matrix<float, 2> new_out_mat1 = out_blob1->get_data()->flatten_2d_matrix();
  EXPECT_EQ(exp_out1, new_out_mat1);

  vsum_layer->backward(input_blob_vec1, need_bp1, output_blob_vec1);
  Matrix<float, 2> exp_diff1 {{1, 2, 3},
                              {1, 2, 3},
                              {1, 2, 3},
                              {1, 2, 3},
                              {1, 2, 3},
                              {1, 3, 5}, 
                              {1, 3, 5}, 
                              {1, 3, 5}, 
                              {1, 3, 5}, 
                              {1, 3, 5}, 
                              {1, 2, 2}, 
                              {1, 2, 2}, 
                              {1, 2, 2}, 
                              {1, 2, 2}, 
                              {1, 2, 2}, 
                              {2, 3, 3},
                              {2, 3, 3},
                              {2, 3, 3},
                              {2, 3, 3},
                              {2, 3, 3}};

  Matrix<float, 2> new_in_diff1 = in_blob1->get_diff()->flatten_2d_matrix();
  EXPECT_EQ(new_in_diff1, exp_diff1);
}


TEST(SoftmaxLayer, forward_backward) {
  LayerParameter lp;
  lp.set_name("softmax1");
  lp.set_type("Softmax_with_loss");
  lp.set_bottom("softmax1_bottom");
  lp.set_top("softmax1_top");
  lp.set_phrase(TRAIN);

  Layer<float> * softmax_with_loss_layer = new SoftmaxWithLossLayer<float>(lp);
  BlobShape in_blob_shape1 {4, 3};
  BlobShape in_blob_shape2 {1, 4};
  BlobShape out_blob_shape {4, 3};

  shared_ptr<Blob<float> > in_blob1 = create_blob_object<float>(in_blob_shape1, true);
  Matrix<float, 2>  in_data_matrix1 = in_blob1->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_data_matrix1 = {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  in_data_matrix1.copy_from(tmp_in_data_matrix1);

  shared_ptr<Blob<float> > in_blob2 = create_blob_object<float>(in_blob_shape2, true);
  Matrix<float, 2>  in_data_matrix2 = in_blob2->get_data()->flatten_2d_matrix();
  Matrix<float, 2> tmp_in_data_matrix2 = {{1, 2, 1, 3}};
  in_data_matrix2.copy_from(tmp_in_data_matrix2);

  shared_ptr<Blob<float> > out_blob = create_blob_object<float>(out_blob_shape, true);

  vector<Blob<float> *>  input_blob_vec;
  vector<Blob<float> *>  output_blob_vec;
  vector<bool> need_bp;
  input_blob_vec.push_back(in_blob1.get());
  input_blob_vec.push_back(in_blob2.get());
  output_blob_vec.push_back(out_blob.get());
  need_bp.push_back(true);

  softmax_with_loss_layer->init(input_blob_vec, output_blob_vec);
  softmax_with_loss_layer->forward(input_blob_vec, output_blob_vec);

  Matrix<float, 2> exp_out 
      {{ 0.09003057,  0.24472847,  0.66524096},
      { 0.01587624,  0.11731043,  0.86681333},
      { 0.1553624,   0.4223188,   0.4223188},
      { 0.1553624,   0.4223188,   0.4223188}};
  Matrix<float, 2> new_out_mat = out_blob->get_data()->flatten_2d_matrix();
  EXPECT_EQ(exp_out, new_out_mat);

  softmax_with_loss_layer->backward(input_blob_vec, need_bp, output_blob_vec);
  Matrix<float, 2> exp_diff 
      {{-0.90996943,  0.24472847,  0.66524096},
 { 0.01587624, -0.88268957,  0.86681333},
 {-0.8446376,   0.4223188,   0.4223188},
 { 0.1553624,   0.4223188,  -0.5776812}};
  Matrix<float, 2> new_in_diff = in_blob1->get_diff()->flatten_2d_matrix();
  EXPECT_EQ(new_in_diff, exp_diff);

}

TEST(NeuralNet, forward_backward) {
    NeuralNet<float> nn;

}
