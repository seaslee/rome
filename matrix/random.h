#ifndef MATRIX_RANDOM_H
#define MATRIX_RANDOM_H

#include <random>
#include "matrix.h"

namespace snoopy {
namespace matrix {

//generate matrix with random values
class Random {
 public:

  //uniform sampling
  template<typename DataType, size_t N>
  static void uniform(Matrix<DataType, N> & t, float a = 0,
                                     float b = 1) {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_real_distribution<> dis(a, b);
    DataType * data = t.get_data()->data();
    for (size_t i = 0; i < t.get_capicity(); ++i) {
      data[i] = dis(gen);
    }
  }

  //guassian sampling
  template<typename DataType, size_t N>
  static void normal(Matrix<DataType, N> & t, float a = 0,
                                    float b = 1) {
    std::random_device r;
    std::mt19937 gen(r());
    std::normal_distribution<> dis(a, b);
    DataType * data = t.get_data()->data();
    for (size_t i = 0; i < t.get_capicity(); ++i) {
      data[i] = dis(gen);
    }
  }

};

}
}

#endif //MATRIX_RANDOM_H

