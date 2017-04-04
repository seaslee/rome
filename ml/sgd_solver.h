#ifndef SNOOPY_ML_SGD_SOLVER_H_
#define SNOOPY_ML_SGD_SOLVER_H_

#include <vector>
#include "nn.h"
#include "solver.h"
#include "../proto/snoopy.pb.h"

//using namespace std;
using std::shared_ptr;
using std::vector;
using std::string;

namespace snoopy {
namespace ml {

template <typename DataType> 
class SGDSolver : public Solver<DataType> {
    public:
        SGDSolver() {}
       ~SGDSolver() {}

       /**
        * create solver from solver parameter
        */
       virtual int init(const SolverParameter & solver);

       /**
        * optimize the neural network
        */
       virtual int update();

       /**
        * save model 
        */
       virtual int save_model();

       /**
        * evaluate model on the test data
        */
       virtual int evaluate_model();

       float get_base_lr() { return base_lr_; }
       float get_momentum() { return momentum_; }

    private:
       shared_ptr<NeuralNet<DataType> > net_;
       float base_lr_;
       float momentum_;
       int max_epochs_;
};

template <typename DataType>
int SGDSolver<DataType>::init(const SolverParameter & solver) {
    net_ = shared_ptr<NeuralNet<DataType> >(new NeuralNet<DataType>);
    NetParameter net_p;
    int status = snoopy::io::read_net_proto_from_text_file(solver.net(), net_p);
    if (status != snoopy::SUCCESS) {
        return snoopy::FAILURE;
    }
    net_->init(net_p);
    base_lr_ = solver.base_lr();
    momentum_ = solver.momentum(); 
    max_epochs_ = solver.epochs();
    return snoopy::SUCCESS;
}

template <typename DataType>
int SGDSolver<DataType>::update() {
    //mini-batch
    DataType loss = static_cast<DataType>(0);
    //create previos blob with momentum
    vector<shared_ptr<Blob<DataType> > > history_param_matrix_vec;
    vector<Blob<DataType> *> para_vector = net_->get_learnable_para_blobs();
    DataFeedLayer<DataType> * data_feed = static_cast<DataFeedLayer<DataType> *>(net_->get_input_feed().get());

    for (int para_index = 0; para_index < para_vector.size(); ++
            para_index) {
        BlobShape blob_shape {static_cast<unsigned long>(para_vector[para_index]->dim_at(0)), 
                                            static_cast<unsigned long>(para_vector[para_index]->dim_at(1))};
       shared_ptr<Blob<DataType> > tmp = create_blob_object<DataType>(
                blob_shape, true);
       Matrix<DataType, 2> history_param_matrix = tmp->get_data()->flatten_2d_matrix();
       Matrix<DataType, 2> derivate_matrix = tmp->get_diff()->flatten_2d_matrix();
       history_param_matrix.clear_data();
       derivate_matrix.clear_data();
       history_param_matrix_vec.push_back(tmp);
    }
    //feed data
    data_feed->read_file();

    for (int epoch_index = 0; epoch_index < max_epochs_; ++epoch_index) {
        cerr << "epoch_index: " << epoch_index << endl;
        for (int iter_index = 0; !data_feed->is_end(); ++iter_index) {
            cerr << "epoch_index: " << epoch_index << " iter_index: " << iter_index << endl;
            data_feed->get_data(net_->get_input_blobs());
            net_->forward(&loss);
            net_->backprop();
            //update the learnabel parameter
            for (int para_index = 0; para_index < para_vector.size(); ++
                    para_index) {
               Matrix<DataType, 2> para_matrix =  para_vector[para_index]->get_data()->flatten_2d_matrix();
               Matrix<DataType, 2> para_diff_matrix =  para_vector[para_index]->get_diff()->flatten_2d_matrix();
               Matrix<DataType, 2> history_param_matrix = history_param_matrix_vec[para_index]->get_data()->flatten_2d_matrix();
               Matrix<DataType, 2> derivate_matrix = history_param_matrix_vec[para_index]->get_diff()->flatten_2d_matrix();

               derivate_matrix = history_param_matrix * momentum_;
               derivate_matrix = derivate_matrix - para_diff_matrix * 
                   (base_lr_ * net_->get_learnable_para_lr()[para_index]);

               para_matrix = para_matrix + derivate_matrix;  
               history_param_matrix.copy_from(derivate_matrix);
            }
        }
        data_feed->clear();
    }

    return snoopy::SUCCESS;
}

template <typename DataType>
int SGDSolver<DataType>::save_model() {
    return snoopy::SUCCESS;
}


template <typename DataType>
int SGDSolver<DataType>::evaluate_model() {
    return snoopy::SUCCESS;
}


}
}

#endif
