#ifndef SNOOPY_IO_TEXT_DATA_H
#define SNOOPY_IO_TEXT_DATA_H

#include "../proto/snoopy.pb.h"
#include "data.h"
#include <cstdlib>
#include <fstream>
#include <vector>
#include "../common/utils.h"
#include "../common/com_def.h"

namespace snoopy {
namespace io{

template <typename DataType>
class TextDataFeed : public DataFeed<DataType> {
    private:
        std::vector<std::vector<std::vector<int> > > data_;     
        int current_index;
        DataFeedParameter data_param_;
    public:
        TextDataFeed(const DataFeedParameter & data_feed_param):
            data_param_(data_feed_param), 
                    current_index(0) {
            }

        void read_file() {
           ifstream in(this->data_param_.filepath().c_str());
           char buffer[1024];
           std::vector<std::string> token_list1;
           std::vector<std::string> token_list2;
           if (!in.is_open()) {
               LOG_FATAL << "Error open filepath " << this->data_param_.filepath();
           }
           std::vector<std::vector<int> > tmp_vec1;
           std::vector<int> tmp_vec2;
           for (int i = 0; i < this->data_param_.max_line() && !in.eof(); ++i) {
               in.getline(buffer, 1024);
               split(buffer, token_list1, ";");
               if (token_list1.size() == 0) {
                   return;
               }
               tmp_vec1.clear();
               for (int j = 0; j < token_list1.size(); ++ j) {
                   tmp_vec2.clear();
                   split(token_list1[j], token_list2, " ");
                   for (int k = 0; k < token_list2.size() && k < this->data_param_.slot_capicity(); ++k) {
                       tmp_vec2.push_back(std::atoi(token_list2[k].c_str()));
                   }
                   tmp_vec1.push_back(tmp_vec2);
               }
               data_.push_back(tmp_vec1);
           }
        }

        virtual int get_data(std::vector<matrix::Blob<DataType> *> & output_blob) {
           CHECK_GE(output_blob.size(), 0);
           CHECK_EQ(output_blob.size(), this->data_param_.slot_size());
           CHECK_EQ(output_blob[0]->dim_at(0), this->data_param_.batch_size());
           CHECK_EQ(output_blob[0]->dim_at(1), this->data_param_.slot_capicity());
           if (data_.size() > 0) {
               CHECK_GE(data_[0].size(), output_blob.size());
               if (data_[0].size() > 0) {
                   CHECK_LE(data_[0][0].size(), output_blob[0]->dim_at(1));
               }
           }

           if (data_.size() - current_index < this->data_param_.batch_size()) {
               return snoopy::SUCCESS;
           }
           for (size_t i = 0; i < this->data_param_.batch_size(); ++i) {
               size_t n_row = current_index + i;
               if (n_row < data_.size()) {
                   for (size_t j = 0; j < this->data_param_.slot_size(); ++j) {
                       for (size_t k = 0; k < this->data_param_.slot_capicity(); ++k) {
                           size_t index_of_blob = k + i * this->data_param_.slot_capicity();
                           if (k < data_[n_row][j].size()) {
                            output_blob[j]->set_data_at(index_of_blob, data_[n_row][j][k]);
                           } else {
                            output_blob[j]->set_data_at(index_of_blob, -1);
                           }
                       }
                   }
               }
           }
           current_index += this->data_param_.batch_size();
           return snoopy::SUCCESS;
        }
};

}
}

#endif
