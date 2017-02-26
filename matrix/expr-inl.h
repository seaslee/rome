/**
 *  Implement the basic algebra operator(+, -, *, /) between matrix and scalar.
 *
 */

#ifndef SNOOPY_MATRIX_EXPR_INL_H
#define SNOOPY_MATRIX_EXPR_INL_H

#include "matrix.h"
#include "matrix_shape.h"
#include "../common/utils.h"

namespace snoopy {
namespace matrix {

/**
 * Check the shape of the matrix for the operation 
 * Make sure the shape of matrix is consistent
 */
template<typename Expr, size_t N>
struct ShapeCheck {
  inline static MatrixShape<N> check(const Expr &e);
};

template<typename DataType, size_t N>
struct ShapeCheck<ScalarExp<DataType>, N> {
  inline static MatrixShape<N> check(const ScalarExp<DataType> &e) {
    MatrixShape<N> s;
    return s;
  }
};

template<typename DataType, size_t N>
struct ShapeCheck<Matrix<DataType, N>, N> {
  inline static MatrixShape<N> check(const Matrix<DataType, N> & e) {
    return e.get_shape();
  }
};

template<typename Op, typename Expr, typename DataType, size_t N>
struct ShapeCheck<SingleOp<Op, Expr, DataType>, N> {
  inline static MatrixShape<N> check(const SingleOp<Op, Expr, DataType> &e) {
    return ShapeCheck<Expr, N>::check(e.left);
  }
};

template<typename Op, typename LeftExpr, typename RightExpr, typename DataType,
    size_t N>
struct ShapeCheck<BinaryOp<Op, LeftExpr, RightExpr, DataType>, N> {
  inline static MatrixShape<N> check(
      const BinaryOp<Op, LeftExpr, RightExpr, DataType> & e) {
    MatrixShape<N> l = ShapeCheck<LeftExpr, N>::check(e.left);
    MatrixShape<N> r = ShapeCheck<RightExpr, N>::check(e.right);
    if (l.shape[0] == 0)
      return r;
    if (r.shape[0] == 0)
      return l;
    check_shape(l == r, "Mismatch matrix shape.");
    return l;
  }
};

/**
 * template function for binary operator
 *
 * With binary_op function, we can transform two matrix to another matrix
 * using a well-defined element-wise uninary operator.
 *  For example:
 *
 *  Matrix<float, 2> a {{2,1}, {3,4}};
 *  Matrix<float, 2> b {{2,2}, {5,4}};
 *  Matrix<float, 2> c(a.shape);
 *  c = single_op<op::add>(a, b);
 *
 *  The result c is :
 *  c = {{4,3}, {8,8}
 *
 *  The algebra operation( +, -, * / ) is implemented using this function.
 */
template<typename Op, typename LeftExpr, typename RightExpr, typename DataType>
inline BinaryOp<Op, LeftExpr, RightExpr, DataType> binary_op(
    const ExprBase<LeftExpr, DataType> &l,
    const ExprBase<RightExpr, DataType>& r) {
  return BinaryOp<Op, LeftExpr, RightExpr, DataType>(l.self(), r.self());
}

/**
 * override operator + for matrix and matrix
 *
 * @param l is left matrix operand
 * @param r is right matrix operand
 * @return the result matrix of (l +  r)
 */
template<typename LeftExpr, typename RightExpr, typename DataType>
inline BinaryOp<op::add, LeftExpr, RightExpr, DataType> operator +(
    const ExprBase<LeftExpr, DataType> &l,
    const ExprBase<RightExpr, DataType>& r) {
  return binary_op<op::add>(l, r);
}

/**
 * override operator - for matrix and matrix
 *
 * @param l is left matrix operand
 * @param r is right matrix operand
 * @return the result matrix of (l - r)
 */
template<typename LeftExpr, typename RightExpr, typename DataType>
inline BinaryOp<op::sub, LeftExpr, RightExpr, DataType> operator -(
    const ExprBase<LeftExpr, DataType> &l,
    const ExprBase<RightExpr, DataType>& r) {
  return binary_op<op::sub>(l, r);
}

/**
 * override operator * for matrix and matrix
 *
 * @param l is left matrix operand
 * @param r is right matrix operand
 * @return the result matrix of (l * r)
 */
template<typename LeftExpr, typename RightExpr, typename DataType>
inline BinaryOp<op::mul, LeftExpr, RightExpr, DataType> operator *(
    const ExprBase<LeftExpr, DataType> &l,
    const ExprBase<RightExpr, DataType>& r) {
  return binary_op<op::mul>(l, r);
}

/**
 * override operator / for matrix and matrix
 *
 * @param l is left matrix operand
 * @param r is right matrix operand
 * @return the result matrix of (l/ r)
 */
template<typename LeftExpr, typename RightExpr, typename DataType>
inline BinaryOp<op::div, LeftExpr, RightExpr, DataType> operator /(
    const ExprBase<LeftExpr, DataType> &l,
    const ExprBase<RightExpr, DataType>& r) {
  return binary_op<op::div>(l, r);
}

/**
 * override operator + for scalar and matrix
 *
 * @param l is left scalar operand
 * @param r is right matrix operand
 * @return the result matrix of (l + r)
 */
template<typename RightExpr, typename DataType>
inline BinaryOp<op::add, ScalarExp<DataType>, RightExpr, DataType> operator +(
    const MATRIX_SCALAR_TYPE_ l, const ExprBase<RightExpr, DataType>& r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(l);
  return binary_op<op::add, ScalarExp<DataType> >(s, r);
}

/**
 * override operator - for scalar and matrix
 *
 * @param l is left scalar operand
 * @param r is right matrix operand
 * @return the result matrix of (l - r)
 */
template<typename RightExpr, typename DataType>
inline BinaryOp<op::sub, ScalarExp<DataType>, RightExpr, DataType> operator -(
    const MATRIX_SCALAR_TYPE_ l, const ExprBase<RightExpr, DataType>& r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(l);
  return binary_op<op::sub, ScalarExp<DataType> >(s, r);
}

/**
 * override operator * for scalar and matrix
 *
 * @param l is left scalar operand
 * @param r is right matrix operand
 * @return the result matrix of (l * r)
 */
template<typename RightExpr, typename DataType>
inline BinaryOp<op::mul, ScalarExp<DataType>, RightExpr, DataType> operator *(
    const MATRIX_SCALAR_TYPE_ l, const ExprBase<RightExpr, DataType>& r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(l);
  return binary_op<op::mul, ScalarExp<DataType> >(s, r);
}

/**
 * override operator / for scalar and matrix
 *
 * @param l is left scalar operand
 * @param r is right matrix operand
 * @return the result matrix of (l / r)
 */
template<typename RightExpr, typename DataType>
inline BinaryOp<op::div, ScalarExp<DataType>, RightExpr, DataType> operator /(
    const MATRIX_SCALAR_TYPE_ l, const ExprBase<RightExpr, DataType>& r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(l);
  return binary_op<op::div, ScalarExp<DataType> >(s, r);
}

/**
 * override operator + for matrix and scalar
 *
 * @param l is matrix operand
 * @param r is right scalar operand
 * @return the result matrix of (l + r)
 */
template<typename LeftExpr, typename DataType>
inline BinaryOp<op::add, LeftExpr, ScalarExp<DataType>, DataType> operator +(
    const ExprBase<LeftExpr, DataType>& l, const MATRIX_SCALAR_TYPE_ r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(r);
  return binary_op<op::add, LeftExpr, ScalarExp<DataType> >(l, s);
}

/**
 * override operator - for matrix and scalar
 *
 * @param l is matrix operand
 * @param r is right scalar operand
 * @return the result matrix of (l - r)
 */
template<typename LeftExpr, typename DataType>
inline BinaryOp<op::sub, LeftExpr, ScalarExp<DataType>, DataType> operator -(
    const ExprBase<LeftExpr, DataType>& l, const MATRIX_SCALAR_TYPE_ r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(r);
  return binary_op<op::sub, LeftExpr, ScalarExp<DataType> >(l, s);
}

/**
 * override operator * for matrix and scalar
 * 
 * @param l is matrix operand
 * @param r is right scalar operand
 * @return the result matrix of (l * r)
 */
template<typename LeftExpr, typename DataType>
inline BinaryOp<op::mul, LeftExpr, ScalarExp<DataType>, DataType> operator *(
    const ExprBase<LeftExpr, DataType>& l, const MATRIX_SCALAR_TYPE_ r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(r);
  return binary_op<op::mul, LeftExpr, ScalarExp<DataType> >(l, s);
}

/**
 * override operator / for matrix and scalar
 *
 * @param l is matrix operand
 * @param r is right scalar operand
 * @return the result matrix of (l / r)
 */
template<typename LeftExpr, typename DataType>
inline BinaryOp<op::div, LeftExpr, ScalarExp<DataType>, DataType> operator /(
    const ExprBase<LeftExpr, DataType>& l, const MATRIX_SCALAR_TYPE_ r) {
  static ScalarExp<DataType> s = ScalarExp<DataType>(r);
  return binary_op<op::div, LeftExpr, ScalarExp<DataType> >(l, s);
}

/**
 * template class for uninary operator
 *
 * With single_op function, we can transform matrix to another matrix
 * using a well-defined element-wise uninary operator.
 *  For example:
 *
 *  Matrix<float, 2> a {{2,1}, {3,4}};
 *  struct square {
 *    template <typename DataType>
 *    inline static DataType matrix_op(const DataType & x) { return x * x; }
 *  }
 *  Matrix<float, 2> b(a.shape);
 *  b = single_op<square>(a);
 *
 *  The result b is :
 *  b = {{4,1}, {9,16}}
 */
template<typename Op, typename Expr, typename DataType>
inline SingleOp<Op, Expr, DataType> single_op(
    const ExprBase<Expr, DataType> &l) {
  return SingleOp<Op, Expr, DataType>(l.self());
}

}  //namespace matrix
} //namespace snoopy

#endif //
