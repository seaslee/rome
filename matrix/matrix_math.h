/**
 *  Complex matrix operation, such as matrix multiplication, etc.
 */

#ifndef SNOOPY_MATRIX_MATH_H_
#define SNOOPY_MATRIX_MATH_H_

#include <vector>
#include <type_traits>
#include "../common/utils.h"
#include "../common/logging.h"

namespace snoopy {
namespace matrix {

/**
 * matrix product function
 *
 * @param dm is the result matrix
 * @param m1 is left operand in the matrix product
 * @param m2 is right operand in the matrix product
 * @param alpha is the scalar to scale
 * @param is_blas: true to use the blas function; otherwise, not
 *
 */
template<typename T1, typename T2, typename DataType, size_t N>
inline void dot(Matrix<DataType, N> & dm, const T1 & m1, const T2 & m2,
                const DataType alpha = 1, const bool is_blas = true) {
  size_t row_prod = m1.get_row();
  size_t column_prod = m2.get_column();
  CHECK_EQ(N, 2);
  CHECK_EQ(m1.get_column(), m2.get_row());
  DataType * l_data = m1.get_data()->data();
  DataType * r_data = m2.get_data()->data();
  DataType * res_data = dm.get_data()->data();
  const size_t l_row = row_prod;
  const size_t l_col = m1.get_column();
  const size_t r_col = column_prod;
  if (is_blas) {
    const int beta = 0;
    const int lda = l_col;
    const int ldb = r_col;
    const int ldc = r_col;
    //call blas matrix-matrix function
#ifdef USE_DOUBLE
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        l_row, r_col, l_col,
        alpha, l_data, lda, r_data, ldb,
        beta, res_data, ldc);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l_row, r_col, l_col,
                alpha, l_data, lda, r_data, ldb, beta, res_data, ldc);
#endif
  } else {
    for (size_t i = 0; i < row_prod; ++i) {
      for (size_t j = 0; j < column_prod; ++j) {
        #ifdef USE_DOUBLE
        res_data[i * column_prod + j] = 0.0;
        #else
        res_data[i * column_prod + j] = 0.0f;
        #endif
        for (size_t k = 0; k < m1.get_column(); ++k) {
          res_data[i * column_prod + j] += l_data[i * l_col + k]
              * r_data[k * r_col + j];
        }
        res_data[i * column_prod + j] *= alpha;
      }
    }
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> dot(const Matrix<DataType, N> & m1,
                               const Matrix<DataType, N> & m2,
                               const DataType alpha = 1, const bool is_blas =
                                   true) {
  size_t row_prod = m1.get_row();
  size_t column_prod = m2.get_column();
  MatrixShape<N> s { row_prod, column_prod };
  Matrix<DataType, N> pr(s);
  dot(pr, m1, m2, alpha, is_blas);
  return pr;
}

/**
 * matrix transpose function
 * @param t is result matrix
 * @param s is input matrix
 * @param alpha is the scalar to scale
 * @param is_blas: true to use the blas function; otherwise, not
 */
template<typename DataType, size_t N, typename T>
inline void transpose(Matrix<DataType, N> & t, const T & s,
                      const DataType alpha = 1, const bool is_blas = false) {
  size_t row = s.get_row();
  size_t column = s.get_column();
  DataType * s_data = s.get_data()->data();
  DataType * t_data = t.get_data()->data();
  if (is_blas) {
//#ifdef USE_DOUBLE
//    cblas_domatcopy(CblasRowMajor, CblasTrans,
//        row, column, alpha, s_data, column, t_data, row); //TODO: test
//#else
//    cblas_somatcopy(CblasRowMajor, CblasTrans,
//        row, column, alpha, s_data, column, t_data, row); //TODO: test
//#endif
  } else {
    for (size_t i = 0; i < column; ++i) {
      for (size_t j = 0; j < row; ++j) {
        t_data[i * row + j] = alpha * s_data[j * column + i];
      }
    }
  }
}

template<typename DataType, size_t N>
inline Matrix<DataType, N> transpose(const Matrix<DataType, N> & s,
                                     const DataType alpha = 1,
                                     const bool is_blas = false) {
  size_t row = s.get_row();
  size_t column = s.get_column();
  MatrixShape<N> sm { column, row };
  Matrix<DataType, N> m(sm, 0);
  transpose(m, s, alpha, is_blas);
  return m;
}

/**
 * Copy data from one matrix to another
 * @param t is destination matrix
 * @param s is source matrix
 */
template<typename DataType, size_t N>
void copyMatrix(Matrix<DataType, N> & t, const Matrix<DataType, N> &s) {
  CHECK_EQ(t.capicity, s.capicity);
  DataType * t_data = t.get_data();
  DataType * s_data = s.get_data();
  for (int i = 0; i < t.getCapicity(); ++i) {
    t_data[i] = s_data[i];
  }
}

/**
 * sum up the matrix
 * @param m is input matrix
 * @param d is sum type,  d= 0: sum to one row, d = 1; sum to one column,
 *        d = 2; sum to a scalar number
 * @return the result matrix
 */
template<typename DataType, size_t N>
Matrix<DataType, 1> sum(const Matrix<DataType, N> &m, int d = 0) {
  CHECK_EQ(N, 2);
  if (d == 0) {
    size_t col = m.get_column();
    MatrixShape<1> sh { col };
    Matrix<DataType, 1> temp(sh);
    for (int i = 0; i < col; ++i) {
      temp[i] = 0;
      for (int j = 0; j < m.get_row(); ++j) {
        temp[i] += m[j][i];
      }
    }
    return temp;
  } else if (d == 1) {
    size_t row = m.get_row();
    MatrixShape<1> sh { row };
    Matrix<DataType, 1> temp(sh);
    for (int i = 0; i < row; ++i) {
      temp[i] = 0;
      for (int j = 0; j < m.get_column(); ++j) {
        temp[i] += m[i][j];
      }
    }
    return temp;
  } else {
    MatrixShape<1> sh { 1 };
    Matrix<DataType, 1> temp(sh);
    temp[0] = 0;
    for (int i = 0; i < m.get_row(); ++i) {
      for (int j = 0; j < m.get_column(); ++j) {
        temp[0] += m[i][j];
      }
    }
    return temp;
  }
}

template<typename DataType>
void sum(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s, int d = 0) {
  if (d == 0) {
    //check column equal
    size_t col_t = t.get_column();
    size_t col_s = s.get_column();
    CHECK_EQ(col_t, col_s);
    for (int i = 0; i < col_t; ++i) {
      t[0][i] = 0;
      for (int j = 0; j < s.get_row(); ++j) {
        t[0][i] += s[j][i];
      }
    }
  } else if (d == 1) {
    //check row equal
    size_t row_t = t.get_row();
    size_t row_s = s.get_row();
    CHECK_EQ(row_t, row_s);
    for (int i = 0; i < row_t; ++i) {
      t[i][0] = 0;
      for (int j = 0; j < s.get_column(); ++j) {
        t[i][0] += s[i][j];
      }
    }
  } 
}

/**
 *  get the max number in the matrix
 * @param m is input matrix
 * @param temp_index is the index of the max value
 * @param d is the way to get the max,  d= 0: maximuim each column to one row;
 *        d = 1: maximuim each row to one column
 *        d = 2; maximuim the matrix
 * @return
 */
template<typename DataType, size_t N>
Matrix<DataType, 1> max(const Matrix<DataType, N> &m, vector<int> &temp_index,
                        int d = 0) {
  CHECK_EQ(N, 2);
  if (d == 0) {
    size_t col = m.get_column();
    MatrixShape<1> sh { col };
    Matrix<DataType, 1> temp(sh);
    for (int i = 0; i < col; ++i) {
      temp[i] = m[0][i];
      temp_index[0] = 0;
      for (int j = 1; j < m.get_row(); ++j) {
        if (temp[i] < m[j][i]) {
          temp[i] = m[j][i];
          temp_index[0] = j;
        }
      }
    }
    return temp;
  } else if (d == 1) {
    size_t row = m.get_row();
    MatrixShape<1> sh { row };
    Matrix<DataType, 1> temp(sh);
    for (int i = 0; i < row; ++i) {
      temp[i] = m[i][0];
      temp_index[i] = 0;
      for (int j = 1; j < m.get_column(); ++j) {
        if (temp[i] < m[i][j]) {
          temp[i] = m[i][j];
          temp_index[i] = j;
        }
      }
    }
    return temp;
  } else {
    MatrixShape<1> sh { 1 };
    Matrix<DataType, 1> temp(sh);
    for (int i = 0; i < m.get_row(); ++i) {
      for (int j = 0; j < m.get_column(); ++j) {
        if (i == 0 && j == 0) {
            temp[0] = m[0][0];
            temp_index[0] = 0;
            temp_index[1] = 0;
        } else if (temp[0] < m[i][j]) {
            temp[0] = m[i][j];
            temp_index[0] = i;
            temp_index[1] = j;
        }
      }
    }
    return temp;
  }
}

template<typename DataType, size_t N>
void oneHotCode(Matrix<DataType, N> &m) {
  CHECK_EQ(N, 2);
  vector<int> ind(m.get_row(), 0);
  Matrix<DataType, 1> tmp = max(m, ind, 1);
  for (int i = 0; i < m.get_row(); ++i) {
    for (int j = 0; j < m.get_column(); ++j) {
      if (j == ind[i]) {
        m[i][j] = 1;
      } else {
        m[i][j] = 0;
      }
    }
  }
}

template<typename DataType, size_t N>
void oneHotCode(Matrix<DataType, N> &m, const vector<int> & v) {
  CHECK_EQ(N, 2);
  for (int i = 0; i < m.get_row(); ++i) {
    for (int j = 0; j < m.get_column(); ++j) {
      if (j == v[i]) {
        m[i][j] = 1;
      } else {
        m[i][j] = 0;
      }
    }
  }
}

/**
 * softmax function
 * @param m is input matrix
 * @return
 */
template<typename DataType, size_t N>
Matrix<DataType, N> softmax(const Matrix<DataType, N> &m) {
  CHECK_EQ(N, 2);
  Matrix<DataType, N> temp(m);
  for (int i = 0; i < m.get_row(); ++i) {
    DataType s = 0;
    DataType ma = m[i][0];
    for (int j = 1; j < m.get_column(); ++j) {
      if (m[i][j] > ma)
        ma = m[i][j];
    }
    for (int j = 0; j < m.get_column(); ++j) {
      s += exp(m[i][j] - ma);
    }
    for (int j = 0; j < m.get_column(); ++j) {
      temp[i][j] = exp(m[i][j] - ma - log(s));
    }
  }
  return temp;
}


template<typename DataType>
void softmax(Matrix<DataType, 2> & t, const Matrix<DataType, 2> &s) {
  CHECK_EQ(s.get_shape(), t.get_shape());
  for (int i = 0; i < s.get_row(); ++i) {
    DataType sum = 0;
    DataType ma = s[i][0];
    for (int j = 1; j < s.get_column(); ++j) {
      if (s[i][j] > ma)
        ma = s[i][j];
    }
    for (int j = 0; j < s.get_column(); ++j) {
      sum += exp(s[i][j] - ma);
    }
    for (int j = 0; j < s.get_column(); ++j) {
      t[i][j] = exp(s[i][j] - ma - log(sum));
    }
  }

}

template<typename DataType>
Matrix<DataType, 2> repmat(const Matrix<DataType, 1> &m, size_t row) {
  MatrixShape<2> s { row, m.get_column() };
  Matrix<DataType, 2> temp(s, 0);
  DataType * t_d = temp.get_data();
  DataType * s_d = m.get_data();
  for (int i = 0; i < row; ++i) {
    std::copy(s_d, s_d + m.get_column(), t_d);
    t_d += m.get_column();
  }
  return temp;
}

template<typename DataType> 
void repmat(Matrix<DataType, 2> & t, const Matrix<DataType, 2> & s, int d) {
    if (d == 0) {
        //repmat the row
        size_t col_t = t.get_column();
        size_t col_s = s.get_column();
        CHECK_EQ(col_t, col_s);
        DataType * t_d = t.get_data()->data();
        DataType * s_d = s.get_data()->data();
        for (int i = 0; i < t.get_row(); ++i) {
            std::copy(s_d, s_d + col_s, t_d);
            t_d += col_t;
        }
    } else if (d == 1) {
        //repmat the column
        size_t row_t = t.get_row();
        size_t row_s = s.get_row();
        size_t col_t = t.get_column();
        CHECK_EQ(row_t, row_s);
        DataType * t_d = t.get_data()->data();
        DataType * s_d = s.get_data()->data();
        for (int i = 0; i < row_t; ++i) {
            std::fill(t_d, t_d + col_t, *s_d);
            t_d += col_t;
            ++s_d;
        }
    }
}

} //namespace matrix
} //namespace snoopy

#endif /* MATRIX_MATH_H_ */
