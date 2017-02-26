#ifndef SNOOPY_IO_TEXT_DATA_H
#define SNOOPY_IO_TEXT_DATA_H

#include "../proto/snoopy.pb.h"
#include "data.h"
#include <fstream>

namespace snoopy {
namespace io{

class TextDataFeed : public DataFeed {
    private:
        vector<vector<vector<int> > > data_;     
        int current_index;
    public:
        TextDataFeed(const DataFeedParameter & data_feed_param):
            data_param_(data_feed_param), 
                    current_index(0) {
            }

        void int read_file() {
           ifstream in(this->data_param_.filepath().c_str());
           char buffer[1024];
           std::vector<std::string> token_list1;
           std::vector<std::string> token_list2;
           if (!in.is_open) {
               LOG(FATAL) << "Error open filepath " << this->data_param_.filepath;
           }
           for (int i = 0; i < this->data_param_.max_line() && !in.eof(); ++i) {
               vector<vector<int> > tmp_vec1;
               in.getline(buffer, 1024);
               split(buffer, token_list1, "\t");
               for (int j = 0; j < token_list1.size(); ++ j) {
                   vector<int> tmp_vec2;
                   split(token_list1[j], token_list2, "\t");
                   for (int k = 0; k < token_list2.size() && k < this->data_param_.slot_capicity(); ++k) {
                       tmp_vec2.push_back(int(token_list2[k]));
                   }
               }
               tmp_vec1.push_back(tmp_vec2);
           }
        }

        virtual int get_data(vector<Blob<DataType> *> & output_blob) {
           CHECK_EQ(output_blob.size(), this->data_param_.slot_size());
           //CHECK_EQ(output_blob[0]->dim_at(0), this->data_param_.slot_capicity()); 
           CHECK_EQ(output_blob[0]->dim_at(1), this->data_param_.batchsize());
           for (int j = 0; j < this->data_param_.slot_size(); ++j) {
               for (int i = 0; i < this->data_param_.batch_size(); ++i) {
                   for (int k = 0; k < this->data_param_.slot_capicity(); ++k) {
                       if (k < data_[current_index + i][j].size()) {
                        output_blob[j][k] = data_[current_index + i][j][k];
                       } else {
                        output_blob[j][k] = -1;
                       }
                   }
               }
           }
           current_index += this->data_param_.batch_size();
           return 0;
        }
};

}
}

#endif
