/**
 *  A template class to describe the shape of Matrix.
 *
 *  This class use the C/C++ row-major multi-array index to organize
 *  the shape for matrix.
 *
 *  Example:
 *    MatrixShape<3> mat_shape {2, 2, 3};
 *    mat_shape is a three-dimension array
 *    MatrixShape<3> mat_shape1 (0, {2, 2, 3});
 *    mat_shape[0] can get the first dimension of the shape
 */

#ifndef SNOOPY_MATRIX_SHAPE_H_
#define SNOOPY_MATRIX_SHAPE_H_

#include <iostream>
#include <memory>
#include <initializer_list>
#include "../common/utils.h"
#include "../common/logging.h"

namespace snoopy {
namespace matrix {

template<size_t N>
struct MatrixShape {
  /**
   * default constructor
   * initialize the shape and stride to zero to get rid of Too Big Memory Allocation.
   */
  MatrixShape()
      : size(0),
        start(0) {
    std::fill(shape, shape + N, 0);
    std::fill(stride, stride + N, 0);
  }
  ;

  /**
   * constructor with initializer list and a start offset
   *
   * @param s is start offset of the matrix
   * @param sp is size of each dimension
   */
  MatrixShape(int s, std::initializer_list<size_t> sp);
  /**
   * constructor with initializer list
   *
   * @param s is start offset of the matrix
   * @param sp is size of each dimension
   * @param st is offset of each dimension
   */
  MatrixShape(int s, std::initializer_list<size_t> sp,
              std::initializer_list<size_t> st);
  /**
   * constructor with initializer list
   *
   * @param sp is size of each dimension
   */
  MatrixShape(std::initializer_list<size_t> sp);
  /**
   * constructor with initializer list
   *
   * @param sp is size of each dimension
   * @param st is offset of each dimension
   */
  MatrixShape(std::initializer_list<size_t> sp,
              std::initializer_list<size_t> st);

  /**
   * copy constructor
   *
   * @param s is another MatrixShape<N> object cloned to this object
   */
  MatrixShape(const MatrixShape<N> &s);
  /**
   * copy assignment operator
   *
   * @param s is another MatrixShape<N> object cloned to this object
   *
   * @return this matrix object that has copied the attributes from another object
   */
  MatrixShape<N> & operator =(const MatrixShape<N> &s);

  /**
   * equal operator
   *
   * @param s is another MatrixShape<N> object compared with this object
   *
   * @return true if equal in all dimension of the ShapeMatrix Object; false,  otherwise
   */
  bool operator ==(const MatrixShape<N> &s);

  /**
   * @return the Sub MatrixShape<N-1> object
   *   For example:
   *    MatrixShape<3> s{2, 3, 4}
   *    s.subShape() return a object is equal to MatrixShape<2> {3,4};
   */
  MatrixShape<N - 1> subShape() const;

  /**
   * index operator
   *
   * @param i is dimension index
   *
   * @return size of the i-th dimension
   */
  size_t & operator [](size_t i) {
    return shape[i];
  }

  /**
   * index operator for const object
   *
   * @param i is dimension index
   *
   * @return const size of the i-th dimension
   */
  const size_t & operator [](size_t i) const {
    return shape[i];
  }
  const int dims = N;
  size_t size; ///< the total size of all dimension
  size_t start; ///< the start offset
  size_t shape[N]; ///< store size of each dimension of the MatrixShape
  size_t stride[N]; ///< store offset of each dimension of the MatrixShape
};


inline void shape2Stride(size_t * stride, size_t * shape, int n) {
  for (int i = 0; i < n; ++i) {
    stride[i] = 1;
    for (int j = i + 1; j < n; ++j) {
      stride[i] *= shape[j];
    }
  }
}

template<size_t N>
inline MatrixShape<N>::MatrixShape(int s, std::initializer_list<size_t> sp) {
  CHECK_EQ(N, sp.size());
  size_t size = 1;
  int i = 0;
  for (auto x : sp) {
    CHECK_GT(x, 0);
    shape[i] = x;
    size *= x;
    i++;
  }
  shape2Stride(stride, shape, N);
  start = s;
}

template<size_t N>
inline MatrixShape<N>::MatrixShape(int s, std::initializer_list<size_t> sp,
                                   std::initializer_list<size_t> st) {
  SN_CHECK(N == sp.size() && N == st.size() && sp.size() == st.size());
  size_t size = 1;
  int i = 0;
  for (auto x : sp) {
    CHECK_GT(x, 0);
    shape[i] = x;
    size *= x;
    i++;
  }
  i = 0;
  for (auto x : st) {
    stride[i] = x;
    i++;
  }
  start = s;
}

template<size_t N>
inline MatrixShape<N>::MatrixShape(std::initializer_list<size_t> sp) {
  //std::cout << N << " " << sp.size() << std::endl;
  CHECK_EQ(N, sp.size());
  size_t size = 1;
  int i = 0;
  for (auto x : sp) {
    CHECK_GT(x, 0);
    shape[i] = x;
    size *= x;
    i++;
  }
  shape2Stride(stride, shape, N);
  start = 0;
}

template<size_t N>
inline MatrixShape<N>::MatrixShape(std::initializer_list<size_t> sp,
                                   std::initializer_list<size_t> st) {
  SN_CHECK(N == sp.size() && N == st.size() && sp.size() == st.size());
  size_t size = 1;
  int i = 0;
  for (auto x : sp) {
    CHECK_GT(x, 0);
    shape[i] = x;
    size *= x;
    i++;
  }
  i = 0;
  for (auto x : st) {
    stride[i] = x;
    i++;
  }
  start = 0;
}

template<size_t N>
inline MatrixShape<N>::MatrixShape(const MatrixShape<N> &s)
    : size(s.size),
      start(s.start) {
  for (int i = 0; i < N; ++i) {
    shape[i] = s.shape[i];
  }
  for (int i = 0; i < N; ++i) {
    stride[i] = s.stride[i];
  }
}

template<size_t N>
inline MatrixShape<N> & MatrixShape<N>::operator =(const MatrixShape<N> &s) {
  if (&s != this) {
    size = s.size;
    start = s.start;
    for (int i = 0; i < N; ++i) {
      shape[i] = s.shape[i];
    }
    for (int i = 0; i < N; ++i) {
      stride[i] = s.stride[i];
    }
  }
  return *this;
}

template<size_t N>
inline bool MatrixShape<N>::operator ==(const MatrixShape<N> &s) {
  for (int i = 0; i < N; ++i) {
    if (shape[i] != s.shape[i])
      return false;
  }
  return true;
}

template<size_t N>
MatrixShape<N - 1> MatrixShape<N>::subShape() const {
  MatrixShape<N - 1> temp;
  temp.start = start;
  temp.size = 1;
  for (int i = 1; i < N; ++i) {
    temp.size *= shape[i];
    temp.shape[i - 1] = shape[i];
  }
  shape2Stride(temp.stride, temp.shape, N - 1);
  return temp;
}

template<size_t N>
inline bool operator==(const MatrixShape<N> & s1,
                        const MatrixShape<N> & s2) {

    for (int i = 0; i < N; ++i) {
        if (s1.shape[i] != s2.shape[i])
          return false;
        }
    return true;
}

} // matrix namespace
} // snoopy namespace

#endif /* MATRIX_SHAPE_H_ */
