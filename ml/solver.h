#ifndef SNOOPY_ML_SOLVER_H_
#define SNOOPY_ML_SOLVER_H_

#include <vector>
#include "nn.h"
#include "../io/get_conf.h"

//using namespace std;
using std::shared_ptr;
using std::vector;
using std::string;

namespace snoopy {
namespace ml {

/**
 * Interface
 */
template <typename DataType>
class Solver {
    public:
        Solver() {}
       ~Solver() {}

       /**
        * create solver from solver parameter
        */
       virtual int init(const SolverParameter & solver) = 0;

       /**
        * optimize the neural network
        */
       virtual int update() = 0;

       /**
        * save model 
        */
       virtual int save_model(const string & save_file_name) {
            io::write_model_to_binary_file(save_file_name, 
                                        net_->get_para_blobs());
            return snoopy::SUCCESS;
       }

       /**
        * evaluate model on the test data
        */
       virtual int evaluate_model(size_t model_index) {
            DataFeedLayer<DataType> * data_feed = static_cast<DataFeedLayer<DataType> *>(this->net_->get_input_feed().get());

           data_feed->clear();
           DataType loss = 0;
           for (int iter_index = 0; !data_feed->is_end(); ++iter_index) {
               data_feed->get_data(this->net_->get_input_blobs());
               this->net_->forward(&loss);
               vector<Blob<DataType> *> output_blobs = net_->get_output_blobs();
               Blob<DataType> * label_blob = net_->get_label_blob();
               if (label_blob == nullptr) {
                    break;
               }
               Matrix<DataType, 2> prob_matrix = output_blobs[0]->get_data()->flatten_2d_matrix();
               Matrix<DataType, 2> label_matrix = label_blob->get_data()->flatten_2d_matrix();
               size_t label_row_n = label_matrix.get_row();

               vector<int> tmp_labels(label_row_n,0);
               Matrix<DataType, 1> prob_max_matrix = max(prob_matrix, tmp_labels, 1);

               for(int i = 0; i < label_row_n; ++i) {
                    probs.push_back(prob_max_matrix[i]);
                    pred_label.push_back(tmp_labels[i]);
                    true_labels.push_back(label_matrix[i][0]);
               }
           }
           int right_n(0);
           for(int i = 0; i < pred_label.size(); ++i) {
              if (pred_label[i] == true_labels[i]) {
                  right_n++;
              } 
           }
           if (pred_label.size() != 0) {
            acc = right_n / float(pred_label.size());
           } else {
            acc = 0;
           }
           LOG_INFO << "Evaluate model " << model_index << " , Acc: " << acc << endl;
           return snoopy::SUCCESS;
       }

    protected:
       shared_ptr<NeuralNet<DataType> > net_;
       vector<float> probs;
       vector<int> pred_label;
       vector<int> true_labels;
       float acc;
       float ppl;
};

}
}

#endif
