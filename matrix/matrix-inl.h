#ifndef MATRIX_INL_H_
#define MATRIX_INL_H_

#include "matrix.h"

namespace snoopy {
namespace matrix {

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(storage::Allocator * a, 
                                   const MatrixShape<N> &s)
    : shape(s),
      stride(s[N - 1]),
      capicity(get_length(s, s[N - 1])),
      data(new storage::Buffer<DataType>(a, capicity)){
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const MatrixShape<N> &s)
    : shape(s),
      stride(s[N - 1]),
      capicity(get_length(s, s[N - 1])),
      data(new storage::Buffer<DataType>(new storage::CPUallocator, capicity)){
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}


template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(storage::TensorBuffer<DataType> * a, const MatrixShape<N> &s, const size_t st)
    : shape(s),
      stride(st),
      capicity(get_length(s, s[N - 1])){ 
  data = a;
  if (data != nullptr) {
     data->ref(); 
  }
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(storage::Allocator * a, 
                                   const MatrixShape<N> &s,
                                   const size_t st)
    : shape(s),
      stride(st),
      capicity(get_length(s, s[N - 1])),
      data(new storage::Buffer<DataType>(a, capicity)){
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(const Matrix<DataType, N> &m) {
    if (data != nullptr) {
        data->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    const Matrix<DataType, N> &m) {
  if (this != &m) {
    //free existing resource
    if (data != nullptr) {
        data->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(Matrix<DataType, N> &&m) {
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
    //copy data
    if (m.get_data() != nullptr) {
        m.get_data()->unref();
    }
    auto m_data_ptr = m.get_data();
    m_data_ptr = nullptr;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    Matrix<DataType, N> &&m) {
  if (this != &m) {
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    //copy data
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
    if (m.get_data() != nullptr) {
        m.get_data()->unref();
    }
    auto m_data_ptr = m.get_data();
    m_data_ptr = nullptr;
  }
  return *this;
}

template<typename DataType, size_t N>
inline int Matrix<DataType, N>::get_capicity() const {
  size_t len = stride;
  for (int i = 0; i < N - 1; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline int Matrix<DataType, N>::get_size() const {
  size_t len = 1;
  for (int i = 0; i < N; ++i) {
    len *= shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline size_t Matrix<DataType, N>::get_length(const MatrixShape<N> &s,
                                              size_t st) const {
  size_t len = st;
  for (int i = 0; i < N - 1; ++i) {
    len *= s.shape[i];
  }
  return len;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N - 1> Matrix<DataType, N>::operator[](
    size_t i) const {
    Matrix<DataType, N-1> sub_matrix;
    sub_matrix.set_shape(shape.subShape());
    sub_matrix.set_stride(shape[N - 1]);
    sub_matrix.set_capicity(sub_matrix.get_length(sub_matrix.get_shape(), sub_matrix.get_stride()));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
    }
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[N - 2]);
    return sub_matrix;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N - 1> Matrix<DataType, N>::operator[](
    size_t i)  {
    Matrix<DataType, N-1> sub_matrix;
    sub_matrix.set_shape(shape.subShape());
    sub_matrix.set_stride(shape[N - 1]);
    sub_matrix.set_capicity(sub_matrix.get_length(sub_matrix.get_shape(), sub_matrix.get_stride()));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    for (int i = 0; i < N - 1; ++i) {
    row *= sub_matrix.get_shape()[i];
    }
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[N - 2]);
    return sub_matrix;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> Matrix<DataType, N>::slice(size_t i,
                                                         size_t j) const {
    Matrix<DataType, N> sub_matrix;
    MatrixShape<N> s(shape);
    s[0] = j - i;
    sub_matrix.set_shape(s);
    sub_matrix.set_stride(s[N - 1]);
    sub_matrix.set_capicity(get_length(s, s[N - 1]));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    for (int i = 0; i < N - 1; ++i) {
        row *= sub_matrix.get_shape()[i];
    }
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[N - 1]);
    return sub_matrix;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> Matrix<DataType, N>::slice(size_t i,
                                                         size_t j) {
    Matrix<DataType, N> sub_matrix;
    MatrixShape<N> s(shape);
    s[0] = j - i;
    sub_matrix.set_shape(s);
    sub_matrix.set_stride(s[N - 1]);
    sub_matrix.set_capicity(get_length(s, s[N - 1]));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    for (int i = 0; i < N - 1; ++i) {
        row *= sub_matrix.get_shape()[i];
    }
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[N - 1]);
    return sub_matrix;
}


template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) += n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) -= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) *= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) /= n;
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator +=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      size_t offset = i * stride + j;
      data->at(offset) += t.get_data()->at(offset);
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator -=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      size_t offset = i * stride + j;
      data->at(offset) -= t.get_data()->at(offset);
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator *=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      size_t offset = i * stride + j;
      data->at(offset) *= t.get_data()->at(offset);
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>& Matrix<DataType, N>::operator /=(
    const Matrix<DataType, N> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      size_t offset = i * stride + j;
      data->at(offset) /= t.get_data()->at(offset);
    }
  }
  return *this;
}

template<typename DataType, size_t N>
inline DataType Matrix<DataType, N>::eval(size_t i, size_t j) const {
  return data->at(i * stride + j);
}

template<typename DataType, size_t N>
inline void Matrix<DataType, N>::set_row_ele(int i,
                                             const Matrix<DataType, N> & s) {
  int row_s = s.get_row();
  int col_s = s.get_column();
  if (row_s == 1) {
    if (column == col_s) {
      for (int j = 0; j < column; ++j) {
        data->at(i * stride + j) = s[0][j];
      }
    } else {
      //std::cerr << "Set Row: Shape not match!" << std::endl;
      LOG_ERROR << "Set Row: Shape not match!";
    }
  } else {
    //std::cerr << "Set Row: Shape not match!" << std::endl;
      LOG_ERROR << "Set Row: Shape not match!";
  }
}

template<typename DataType, size_t N>
template<typename SubType>
inline Matrix<DataType, N>& Matrix<DataType, N>::operator=(
    const ExprBase<SubType, DataType> &e) {
  const SubType & sub = e.self();
  ShapeCheck<SubType, N>::check(sub);
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) = sub.eval(i, j);
    }
  }
  return *this;
}

template<size_t N, typename List, typename L>
typename std::enable_if<(N == 1), void>::type fill_shape(const List & t,
                                                         L * shape) {
  *shape = t.size();
}

template<size_t N, typename List, typename L>
typename std::enable_if<(N > 1), void>::type fill_shape(const List & t,
                                                        L * shape) {
  auto i = t.begin();
  for (auto j = i + 1; j != t.end(); ++j) {
    /*if (i->size() != j->size()) {
      std::cerr << "Row size inconsistent" << std::endl;
      exit(1);
    }*/
    CHECK_EQ(i->size(), j->size());
  }
  *shape++ = t.size();
  fill_shape<N - 1>(*t.begin(), shape);
}

template<typename List, size_t N>
void init_shape(const List &l, MatrixShape<N> &ss) {
  //std::array<size_t, N> a;
  auto s = ss.shape;
  fill_shape<N>(l, s);
  size_t size = 1;
  for (int i = 0; i < N; ++i) {
    size *= s[i];
  }
  shape2Stride(ss.stride, ss.shape, N);
  ss.start = 0;
}

template<typename T, typename DataType>
void fill_ele(const T* s, const T* e, storage::TensorBuffer<DataType> * data, 
        size_t &offset) {
  for (auto i = s; i != e; ++i) {
    data->at(offset++) = *i;
  }
}

template<typename T, typename DataType>
void fill_ele(const std::initializer_list<T>* s,
              const std::initializer_list<T>* e, 
              storage::TensorBuffer<DataType> * data,
              size_t &offset) {
  for (; s != e; ++s) {
    fill_ele(s->begin(), s->end(), data, offset);
  }
}

template<typename T, typename DataType>
void init_ele(std::initializer_list<T> l, storage::TensorBuffer<DataType> * d, size_t & offset) {
  fill_ele(l.begin(), l.end(), d, offset);
}

template<typename DataType, size_t N>
inline Matrix<DataType, N>::Matrix(matrix_initializer_list<DataType, N> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[N - 1];
  capicity = get_capicity();
  data = new storage::Buffer<DataType>(new storage::CPUallocator, capicity);
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> & Matrix<DataType, N>::operator =(
    matrix_initializer_list<DataType, N> t) {
  init_shape(t, shape);
  //shape = s;
  stride = shape[N - 1];
  capicity = get_capicity();
  data = new storage::Buffer<DataType>(new storage::CPUallocator, capicity);
  row = 1;
  for (int i = 0; i < N - 1; ++i) {
    row *= shape[i];
  }
  column = shape[N - 1];
  size_t offset = 0;
  init_ele(t, data, offset);
  return *this;
}

//copy from source matrix
template <typename DataType, size_t N>
inline void Matrix<DataType,N>::copy_from(const Matrix<DataType, N> &s) {
    CHECK_GE(capicity, s.get_capicity());
    shape = s.get_shape();
    stride = s.get_stride();
    row = s.get_row();
    column = s.get_column();
    std::copy(s.get_data()->data(), s.get_data()->data() + s.get_size(), data->data());
}

template <typename DataType, size_t N>
inline void Matrix<DataType,N>::clear_data() {
    std::fill(data->data(), data->data() + capicity, 0);
}

template <typename DataType>
inline void Matrix<DataType,1>::copy_from(const Matrix<DataType, 1> &s) {
    CHECK_GE(capicity, s.get_capicity());
    shape = s.get_shape();
    stride = s.get_stride();
    row = s.get_row();
    column = s.get_column();
    std::copy(s.get_data()->data(), s.get_data()->data() + s.get_size(), data->data());
}

template <typename DataType>
inline void Matrix<DataType,1>::clear_data() {
    std::fill(data->data(), data->data() + capicity, 0);
}



//one-dimension matrix implementation
template<typename DataType>
inline Matrix<DataType, 1>::Matrix(storage::Allocator * a, 
                                   const MatrixShape<1> &s)
    : shape(s),
      stride(s[0]),
      capicity(get_length(s, s[0])),
      data(new DataType[capicity]),
      row(1),
      column(shape[0]) {
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(storage::Allocator * a, 
                                   const MatrixShape<1> &s,
                                   const size_t st)
    : shape(s),
      stride(st),
      capicity(get_length(s, s[0])),
      data(new DataType[capicity]),
      row(1),
      column(shape[0]) {
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const MatrixShape<1> &s)
    : shape(s),
      stride(s[0]),
      capicity(get_length(s, s[0])),
      data(new storage::Buffer<DataType>(new storage::CPUallocator, capicity)),
      row(1),
      column(s[0]) {
}


template<typename DataType>
inline Matrix<DataType, 1>::Matrix(const Matrix<DataType, 1> &m) {
    //free existing resource
    if (data != nullptr) {
        data->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
}


template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    const Matrix<DataType, 1> &m) {
  if (this != &m) {
    //free existing resource
    if (data != nullptr) {
        data->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(Matrix<DataType, 1> &&m) {
    //copy data
    if (m.get_data() != nullptr) {
        m.get_data()->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
    auto m_data_ptr = m.get_data();
    m_data_ptr = nullptr;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    Matrix<DataType, 1> &&m) {
  if (this != &m) {
    //copy data
    if (m.get_data() != nullptr) {
        m.get_data()->unref();
    }
    shape = m.get_shape();
    stride = m.get_stride();
    capicity = get_length(m.get_shape(), m.get_stride());
    row = m.row;
    column = m.column;
    data = m.get_data();
    if (data != nullptr) {
        data->ref();
    }
    auto m_data_ptr = m.get_data();
    m_data_ptr = nullptr;
  }
  return *this;
}
template<typename DataType>
inline int Matrix<DataType, 1>::get_capicity() const {
  return stride;
}

template<typename DataType>
inline int Matrix<DataType, 1>::get_size() const {
  return shape[0];
}

template<typename DataType>
inline size_t Matrix<DataType, 1>::get_length(const MatrixShape<1> &s,
                                              size_t stride) const {
  return stride;
}

template<typename DataType>
inline const DataType & Matrix<DataType, 1>::operator[](size_t i) const {
  return data->at(i);  
}
template<typename DataType>
inline DataType & Matrix<DataType, 1>::operator[](size_t i) {
  return data->at(i);
}

template<typename DataType>
inline Matrix<DataType, 1> Matrix<DataType, 1>::slice(size_t i,
                                                         size_t j) const {
    Matrix<DataType, 1> sub_matrix;
    MatrixShape<1> s(shape);
    s[0] = j - i;
    sub_matrix.set_shape(s);
    sub_matrix.set_stride(s[0]);
    sub_matrix.set_capicity(get_length(s, s[0]));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[0]);

    return sub_matrix;
}

template<typename DataType>
inline Matrix<DataType, 1> Matrix<DataType, 1>::slice(size_t i,
                                                         size_t j) {
    Matrix<DataType, 1> sub_matrix;
    MatrixShape<1> s(shape);
    s[0] = j - i;
    sub_matrix.set_shape(s);
    sub_matrix.set_stride(s[0]);
    sub_matrix.set_capicity(get_length(s, s[0]));
    sub_matrix.set_data(new storage::SubBuffer<DataType>(data, i * shape.stride[0]));
    size_t row = 1;
    sub_matrix.set_row(row);
    sub_matrix.set_column(sub_matrix.get_shape()[0]);
    return sub_matrix;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator +=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) += n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator -=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) -= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator *=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) *= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator /=(
    const DataType & n) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) /= n;
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator +=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) += t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator -=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) -= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator *=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) *= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1>& Matrix<DataType, 1>::operator /=(
    const Matrix<DataType, 1> & t) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) /= t[i][j];
    }
  }
  return *this;
}

template<typename DataType>
inline DataType Matrix<DataType, 1>::eval(size_t i, size_t j) const { /*broadcasting the matrix*/
  return data->at(i * stride + j);
}

template<typename DataType>
inline void Matrix<DataType, 1>::set_row_ele(int i,
                                             const Matrix<DataType, 1> & s) {
  int row_s = s.get_row();
  int col_s = s.get_column();
  if (row_s == 1) {
    if (column == col_s) {
      for (int j = 0; j < column; ++j) {
        data->at(i * stride + j) = s[0][j];
      }
    } else {
      //std::cerr << "Set Row: Shape not match!" << std::endl;
      LOG_ERROR << "Set Row: Shape not match!";
    }
  } else {
    //std::cerr << "Set Row: Shape not match!" << std::endl;
      LOG_ERROR << "Set Row: Shape not match!";
  }
}

template<typename DataType>
template<typename SubType>
inline Matrix<DataType, 1>& Matrix<DataType, 1>::operator=(
    const ExprBase<SubType, DataType> &e) {
  const SubType & sub = e.self();
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < column; ++j) {
      data->at(i * stride + j) = sub.eval(i, j);
    }
  }
  return *this;
}

template<typename DataType>
inline Matrix<DataType, 1>::Matrix(matrix_initializer_list<DataType, 1> t) {
  init_shape(t, shape);
  stride = shape[0];
  capicity = get_capicity();
  data = new storage::Buffer<DataType>(new storage::CPUallocator, capicity);
  row = 1;
  column = shape[0];
  size_t offset = 0;
  init_ele(t, data, offset);
}

template<typename DataType>
inline Matrix<DataType, 1> & Matrix<DataType, 1>::operator =(
    matrix_initializer_list<DataType, 1> t) {
  init_shape(t, shape);
  stride = shape[0];
  capicity = get_capicity();
  data = new storage::Buffer<DataType>(new storage::CPUallocator, capicity);
  row = 1;
  column = shape[0];
  size_t offset = 0;
  init_ele(t, data, offset);
  return *this;
}

template<typename DataType, size_t N>
std::ostream & operator <<(std::ostream &os, const Matrix<DataType, N> & m) {
  for (size_t i = 0; i < m.get_capicity(); ++i) {
      os << m.get_data()->data()[i] << " ";
  }
  os << std::endl;
  return os;
}

}
}
#endif /* MATRIX_INL_H_ */
