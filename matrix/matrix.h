/**
 *  A template class to implement matrix and support element-wise operation on it
 *
 *  This class implements the template matrix. Matrix object can be constructed 
 *  by the following ways:
 *
 *  a. a matrix can be constructed with the matrix_initializer_list (
 *  a.k, recursive initializer_list). The memory of matrix is controlled by a default
 *  memory object.
 *
 *  For example:
 *      Matrix<float, 2> amat {{1,2}, {2, 4}};
 *
 *  b. the matrix can also constructed with a memory allocator and the shape of matrix. 
 *
 *  For example:
 *      storage::Allocator * a = new storage::CPUallocator;  
 *      Matrix<float 2> rmat(a, {2,2});
 *  Note that the memory is uninitialized.
 *
 *  c. the matrix can also be gotten from copy constructor and copy assignment. The new 
 *  matrix is shared the memory with the origin matrix.
 *
 */
#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <fstream>
#include <cblas.h>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <array>
#include "expr.h"
#include "matrix_shape.h"
#include "../storage/buffer.h"

//#define DEBUG
namespace snoopy {
namespace matrix {

// recursive initializer_list for Matrix Class
template<typename DataType, size_t N>
struct MatrixInit {
  using type = std::initializer_list<typename MatrixInit<DataType, N-1>::type>;
};

template<typename DataType>
struct MatrixInit<DataType, 1> {
  using type = std::initializer_list<DataType>;
};

template<typename DataType>
struct MatrixInit<DataType, 0>;

template<typename DataType, size_t N>
using matrix_initializer_list = typename MatrixInit<DataType, N>::type;

template<typename DataType, size_t N>
class Matrix : public ExprBase<Matrix<DataType, N>, DataType> {
 public:
  /** default constructor
   */
  Matrix()
      : data(nullptr),
        stride(size_t(0)),
        capicity(size_t(0)),
        row(size_t(0)),
        column(size_t(0)) {
  }
  ;

  /** 
   * constructor with a memory allocator and matrix shape
   *
   * @param a: allocate the memory 
   * @param s: shape of matrix
   */
  Matrix(storage::Allocator * a, const MatrixShape<N> &s);

  /** 
   * constructor with matrix shape
   *
   * @param a: allocate the memory 
   * @param s: shape of matrix
   */
  Matrix(const MatrixShape<N> &s);


  /**
   * constructor with a memory allocator, matrix shape and stride
   *
   * @param a: buffer storage
   * @param s: shape of the matrix
   * @param st: stride for each row of the multi-array
   * Note that align memory and change the shape[N-1] memeory
   */
  Matrix(storage::TensorBuffer<DataType> * a, const MatrixShape<N> &s, const size_t st);

  /**
   * constructor with a memory allocator, matrix shape and stride
   *
   * @param a: allocate the memory 
   * @param s: shape of the matrix
   * @param st: stride for each row of the multi-array
   * Note that align memory and change the shape[N-1] memeory
   */
  Matrix(storage::Allocator * a, const MatrixShape<N> &s, const size_t st);

  /**
   * copy constructor
   * 
   * @param m: another matrix to be cloned to this matrix
   *
   * @return the matrix that has copied data from another matrix
   */
  Matrix(const Matrix<DataType, N> &m);

  /**
   * copy assignment
   *
   * @param m: another matrix to be cloned to this matrix
   *
   * @return the matrix that has copied data from another matrix
   */
  Matrix<DataType, N> & operator =(const Matrix<DataType, N> &m);

  /**
   * move copy constructor
   *
   * @param m: another matrix to be cloned to this matrix
   *
   * @return the matrix that has copied data from another matrix
   */
  Matrix(Matrix<DataType, N> &&m);

  /**
   * move assignment operator
   *
   * @param m: matrix to be moved to this matrix
   *
   * @return the matrix that has gotten data from another matrix
   *
   */
  Matrix<DataType, N> & operator =(Matrix<DataType, N> &&m);

  /**
   * constructor with matrix_initializer_list
   *
   * @param t is initializer_list for the matrix
   */
  Matrix(matrix_initializer_list<DataType, N> t);

  /**
   * assignment operator with matrix_initializer_list
   *
   * @param t is initializer_list for the matrix
   *
   * @return the matrix that has filled the data with the initializer_list
   */
  Matrix & operator =(matrix_initializer_list<DataType, N> t);

  template<typename T>
  Matrix(std::initializer_list<T> t) = delete;

  template<typename T>
  Matrix & operator =(std::initializer_list<T> t) = delete;

  /**
   * de-constructor
   */
  ~Matrix() {
    if (data) {
        data->unref();
    }
  }

  /**
   * index operation for the matrix
   *
   * @param i is the index
   *
   * @return a Matrix Object which is refer to some elements in the Matrix Object
   */
  inline Matrix<DataType, N - 1> operator[](size_t i) const;
  inline Matrix<DataType, N - 1> operator[](size_t i);

  /**
   * slice function for the matrix
   *
   * @param i is the start index
   * @param j is the end index
   *
   * @return a SubMatrix Object which is refer to some elements in the Matrix Object
   */
  inline Matrix<DataType, N> slice(size_t i, size_t j) const;
  inline Matrix<DataType, N> slice(size_t i, size_t j);

  /**
   * Scalar add Operator
   *
   * @param n is the scalar added to matrix
   *
   * @return the matrix in which the elements are all added the scalar value
   */
  inline Matrix<DataType, N> & operator +=(const DataType & n);

  /**
   * Scalar sub Operator
   *
   * @param n is the scalar subscribed to matrix
   *
   * @return the matrix in which the elements are all subscribed the scalar value
   */
  inline Matrix<DataType, N> & operator -=(const DataType & n);

  /**
   * Scalar multiply Operator
   *
   * @param n is the scalar multiplied to matrix
   *
   * @return the matrix in which the elements are all multiplied the scalar value
   */
  inline Matrix<DataType, N> & operator *=(const DataType & n);

  /**
   * Scalar devision Operator
   *
   * @param n is the scalar divided to matrix
   *
   * @return the matrix in which the elements are all devided the scalar value
   */
  inline Matrix<DataType, N> & operator /=(const DataType & n);

  /**
   * add Operator
   *
   * @param t is the matrix added to this matrix
   *
   * @return the matrix in which the elements: m1(i,j) += m2(i,j)
   */
  inline Matrix<DataType, N> & operator +=(const Matrix<DataType, N> & t);

  /**
   * sub Operator
   *
   * @param t is the matrix subscribed to this matrix
   *
   * @return the matrix in which the elements: m1(i,j) -= m2(i,j)
   */
  inline Matrix<DataType, N> & operator -=(const Matrix<DataType, N> & t);

  /**
   * multiply Operator
   *
   * @param t is the matrix multiplied to this matrix
   *
   * @return the matrix in which the elements: m1(i,j) *= m2(i,j)
   */
  inline Matrix<DataType, N> & operator *=(const Matrix<DataType, N> & t);

  /**
   * division opertor
   *
   * @param t is the matrix divided to this matrix
   *
   * @return the matrix in which the elements: m1(i,j) /= m2(i,j)
   */
  inline Matrix<DataType, N>& operator /=(const Matrix<DataType, N> & t);

  /**
   * assign Operator
   *
   * @param e is the right matrix object or a scalar
   *
   * @return the result matrix
   */
  template<typename SubType>
  inline Matrix<DataType, N>& operator=(const ExprBase<SubType, DataType> &e);

  /**
   * get the basic element in index (i,j)
   *
   * @param i is the row index
   * @param j is the column index
   *
   * @return the element in the index(i,j)
   */
  inline DataType eval(size_t i, size_t j) const;

  inline void set_row_ele(int i, const Matrix<DataType, N> & s);

  inline size_t get_column() const {
    return column;
  }

  inline size_t get_row() const {
    return row;
  }

  inline int get_capicity() const;
  inline int get_size() const;

  /**
   * get the length of the shape
   * @param s is the shape object
   * @param stride is the row length
   * @return the length of the shape with the stride
   */
  inline size_t get_length(const MatrixShape<N> &s, size_t stride) const;
  inline storage::TensorBuffer<DataType>  * get_data() {
    return data;
  }
  inline  storage::TensorBuffer<DataType>  * get_data() const {
    return data;
  }

  inline MatrixShape<N> get_shape() const { return shape; }
  inline size_t get_stride() const {return stride;}
  inline size_t get_stride()  {return stride;}
  inline void set_shape(const MatrixShape<N> & s) { shape = s; }
  inline void set_stride(const size_t s) { stride = s; }
  inline void set_capicity(const size_t c) { capicity = c;}
  inline void set_data(storage::TensorBuffer<DataType> * d) { data = d; }
  inline void set_row(const size_t r) {row = r;}
  inline void set_column(const size_t c) { column = c; }
  inline void get_size() {return row * column;};

  /*
   * copy data from source matrix to this matrix
   * make sure the capicity larger than the source matrix 
   *
   * param s: source target
   */
  inline void copy_from(const Matrix<DataType, N> & s);
  /**
   * set data field to zero
   */
  inline void clear_data();

  private:
  MatrixShape<N> shape;
  size_t stride;
  size_t capicity;
  storage::TensorBuffer<DataType> * data;
  size_t row;
  size_t column;
};

/**
 * The matrix for the specified one dimension matrix(vector)
 *
 */
template<typename DataType>
class Matrix<DataType, 1> : public ExprBase<Matrix<DataType, 1>, DataType> {
 public:
  Matrix()
      : data(nullptr),
        stride(size_t(0)) {
  }
  ;
  Matrix(storage::Allocator * a, const MatrixShape<1> &s);

  Matrix(storage::Allocator * a, const MatrixShape<1> &s, const size_t st);

  Matrix(const MatrixShape<1> &s);

  Matrix(const MatrixShape<1> &s, const size_t st);

  Matrix(const Matrix<DataType, 1> &m);
  Matrix<DataType, 1> & operator =(const Matrix<DataType, 1> &m);

  Matrix(Matrix<DataType, 1> &&m);
  Matrix<DataType, 1> & operator =(Matrix<DataType, 1> &&m);

  Matrix(matrix_initializer_list<DataType, 1> t);
  Matrix & operator =(matrix_initializer_list<DataType, 1> t);
  ~Matrix() {
      if (data) {
        data->unref();
      }
  }

  int get_capicity() const;
  int get_size() const;
  size_t get_length(const MatrixShape<1> &s, size_t stride) const;
  storage::TensorBuffer<DataType> * get_data() {
    return data;
  }

  storage::TensorBuffer<DataType> * get_data() const {
    return data;
  }
  const DataType & operator[](size_t i) const;
  DataType & operator[](size_t i);

  Matrix<DataType, 1> slice(size_t i, size_t j) const;
  Matrix<DataType, 1> slice(size_t i, size_t j);
  Matrix<DataType, 1> & operator +=(const DataType & n);
  Matrix<DataType, 1> & operator -=(const DataType & n);
  Matrix<DataType, 1> & operator *=(const DataType & n);
  Matrix<DataType, 1> & operator /=(const DataType & n);
  Matrix<DataType, 1> & operator +=(const Matrix<DataType, 1> & t);
  Matrix<DataType, 1> & operator -=(const Matrix<DataType, 1> & t);
  Matrix<DataType, 1> & operator *=(const Matrix<DataType, 1> & t);
  Matrix<DataType, 1>& operator /=(const Matrix<DataType, 1> & t);

  template<typename SubType>
  Matrix<DataType, 1>& operator=(const ExprBase<SubType, DataType> &e);

  DataType eval(size_t i, size_t j) const;
  void set_row_ele(int i, const Matrix<DataType, 1> & s);
  size_t get_column() const {
    return column;
  }

  size_t get_row() const {
    return row;
  }

  inline size_t get_stride() const {return stride;}
  inline size_t get_stride()  {return stride;}

  inline MatrixShape<1> get_shape() const {return shape;}

  inline void set_shape(const MatrixShape<1> & s) { shape = s; }
  inline void set_stride(const size_t s) { stride = s; }
  inline void set_capicity(const size_t c) { capicity = c;}
  inline void set_data(storage::TensorBuffer<DataType> * d) { data = d; }
  inline void set_row(const size_t r) {row = r;}
  inline void set_column(const size_t c) { column = c; }
  inline void copy_from(const Matrix<DataType, 1> & s);
  inline void clear_data();

  private:
  MatrixShape<1> shape;
  size_t stride;
  size_t capicity;
  storage::TensorBuffer<DataType> * data;
  size_t row;
  size_t column;
};

template<typename DataType>
inline bool float_equal(const DataType& f1, const DataType& f2) {
    if (std::fabs(f1-f2) <= 1e-6) {
        return true;
    }
    return false;
}
/**
 * Ruturn true if all elements in the matrix is equal and stride is equal
 *
 */
template<typename DataType, size_t N>
inline bool operator==(const Matrix<DataType, N>& m1,
                       const Matrix<DataType, N>& m2) {
  storage::TensorBuffer<DataType> * d1 = m1.get_data();
  storage::TensorBuffer<DataType> * d2 = m2.get_data();
  if (m1.get_size() != m2.get_size())
    return false;
  for (size_t i = 0; i < m1.get_size(); ++i) {
    if (!float_equal(d1->at(i), d2->at(i)))
      return false;
  }
  return true;
}

}  //namespace matrix
}  //namespace snoopy

#ifdef MATRIX_SCALAR_TYPE_
#error "MATRIX_SCALAR_TYPE_ Should not be defined!"
#endif
#define MATRIX_SCALAR_TYPE_ float
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#define MATRIX_SCALAR_TYPE_ double
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#define MATRIX_SCALAR_TYPE_ int
#include "expr-inl.h"
#undef MATRIX_SCALAR_TYPE_
#include "matrix-inl.h"
#include "matrix_math.h"
#include "random.h"
#endif // MATRIX_MATRIX_H

