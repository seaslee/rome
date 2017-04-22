#ifndef SNOOPY_ML_TEXT_DATA_H
#define SNOOPY_ML_TEXT_DATA_H


#include "layer.h"
#include "data_layer.h"

namespace snoopy {
namespace ml{

template <typename DataType>
class TextDataFeedLayer : public DataFeedLayer<DataType> {
    private:
        std::vector<std::vector<std::vector<int> > > data_;     
        int current_index;
        DataFeedParameter data_param_;
        bool is_end_;
    public:
     explicit TextDataFeedLayer(const LayerParameter & para) :
         DataFeedLayer<DataType>(para) {}
        virtual void init_spec_layer(const vector<Blob<DataType> *> & input_blob,
                    const vector<Blob<DataType> *> & output_blob);

        ~TextDataFeedLayer() {}

        virtual void clear() {
            is_end_ = false;
            current_index = 0;
        }

        virtual bool is_end() {
           return is_end_; 
        }

        virtual void read_file();

        virtual int get_data(std::vector<matrix::Blob<DataType> *> & output_blob);
        inline virtual int exact_bottom_blob() {return 0;}
        inline virtual int exact_top_blob() {return data_param_.slot_size(); }
};

}
}

#endif
