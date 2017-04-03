#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h> 
#include "../proto/snoopy.pb.h"
#include "../io/get_conf.h"

using namespace snoopy::io;
using namespace snoopy;

int main(int argc, char** argv) { 
    testing::InitGoogleTest(&argc, argv); 
    // Runs all tests using Google Test. 
    return RUN_ALL_TESTS(); 
}

TEST(read_net_proto, get_data) {
    NetParameter net_p;
    int status = read_net_proto_from_text_file("./net_demo.proto.txt", net_p);
    EXPECT_EQ(status, snoopy::SUCCESS);
}
