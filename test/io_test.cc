#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h> 
#include "../proto/snoopy.pb.h"
#include "../io/text_data.h"

using namespace snoopy::io;
using namespace snoopy;
using namespace snoopy::matrix;

int main(int argc, char** argv) { 
    testing::InitGoogleTest(&argc, argv); 
    // Runs all tests using Google Test. 
    return RUN_ALL_TESTS(); 
}

TEST(TextDataFeed, get_data) {
    DataFeedParameter dp;
    dp.set_filepath("./data.txt");
    dp.set_slot_capicity(5);
    dp.set_slot_size(3);
    dp.set_batch_size(4);
    dp.set_max_line(1024);
    dp.add_t_blob_name("slot1");
    dp.add_t_blob_name("slot2");
    dp.add_t_blob_name("slot3");

    TextDataFeed<float> feed(dp);
    feed.read_file();
    BlobShape out_blob_shape {static_cast<unsigned long>(dp.batch_size()), 
                            static_cast<unsigned long>(dp.slot_capicity())};
    shared_ptr<Blob<float> > out_blob1 = create_blob_object<float>(out_blob_shape, true);
    shared_ptr<Blob<float> > out_blob2 = create_blob_object<float>(out_blob_shape, true);
    shared_ptr<Blob<float> > out_blob3 = create_blob_object<float>(out_blob_shape, true);
    vector<Blob<float> *>  output_blob_vec;
    output_blob_vec.push_back(out_blob1.get());
    output_blob_vec.push_back(out_blob2.get());
    output_blob_vec.push_back(out_blob3.get());
    feed.get_data(output_blob_vec);
    Matrix<float, 2> exp_out1 {{1,2,3,-1,-1},
                              {1,2,3,-1,-1},
                              {1,2,3,-1,-1},
                              {1,2,3,-1,-1}};
    Matrix<float, 2> exp_out2 {{4,5,6,-1,-1},
                              {4,5,6,-1,-1},
                              {4,5,6,-1,-1},
                              {4,5,6,-1,-1}};
    Matrix<float, 2> exp_out3 {{1,-1, -1,-1,-1},
                              {1,-1,-1,-1,-1},
                              {1,-1,-1,-1,-1},
                              {1,-1,-1,-1,-1}};
     Matrix<float, 2> out1 = output_blob_vec[0]->get_data()->flatten_2d_matrix();
     Matrix<float, 2> out2 = output_blob_vec[1]->get_data()->flatten_2d_matrix();
     Matrix<float, 2> out3 = output_blob_vec[2]->get_data()->flatten_2d_matrix();
     EXPECT_EQ(out1, exp_out1);
     EXPECT_EQ(out2, exp_out2);
     EXPECT_EQ(out3, exp_out3);
}

