/**
 *  \file expr.h
 *  \brief Define the expression base class and the derived expression Class
 *         for single operator and binary operator.
 *  \author wbd
 *
 *  The expression template trick is used to make the algebra operation easy.
 *  For example, with the expression template, the expression Matd = Mata + Matb + Matc
 *  can be transformed by c++ compile to the following:
 *  for(int i=0; i<row; ++i)
 *    for(int j=0; j<column; ++j)
 *      Matd[i][j] = Mata[i][j] + Matb[i][j] + Matc[i][j]
 *  So we don't have to write the complex loop code for every matrix. However, we also
 *  can get the fast running speed.
 */

#ifndef SNOOPY_MATRIX_EXPR_H
#define SNOOPY_MATRIX_EXPR_H
namespace snoopy {
namespace matrix {

namespace op {

/**
 * basic binary algebra operation +
 *
 * Define the + operation for the binary expression. The operator can
 * be passed to the single and binary expression as the template argument.
 * Note that:
 * It should define a static method called "matrix_op". For binary expression,
 * the matrix_op is the following format:
 *   inline static DataType matrix_op(const DataType & , const DataType & );
 * For the single expression, the matrix_op is the following format:
 *   inline static DataType matrix_op(const DataType & );
 */
struct add {
  template<typename DataType>
  inline static DataType matrix_op(const DataType & l, const DataType & r) {
    return l + r;
  }
};

struct sub {
  template<typename DataType>
  inline static DataType matrix_op(const DataType & l, const DataType & r) {
    return l - r;
  }
};

struct mul {
  template<typename DataType>
  inline static DataType matrix_op(const DataType & l, const DataType & r) {
    return l * r;
  }
};

struct div {
  template<typename DataType>
  inline static DataType matrix_op(const DataType & l, const DataType & r) {
    return l / r;
  }
};

}  //namespace op

/**
 * Expression template base class
 *
 * All sub class should put their type in place "SubExpr"
 */
template<typename SubExpr, typename DataType>
class ExprBase {
 public:
  inline const SubExpr & self(void) const {
    return *(static_cast<const SubExpr *>(this));
  }
};

/**
 * ScalarExp represent the scalar expression
 *
 * With the ScalarExp, we can write the matrix opertion with scalar,
 * such as:  a * Matrix, where "a" is a scalar.
 */
template<typename DataType>
class ScalarExp : public ExprBase<ScalarExp<DataType>, DataType> {
 public:
  const DataType s_val;
  inline ScalarExp() {
  }
  ;
  inline ScalarExp(const ScalarExp & s)
      : s_val(s.s_val) {
  }
  ;
  inline ScalarExp(const DataType &s)
      : s_val(s) {
  }
  ;
  inline const DataType eval(size_t i, size_t j) const {
    return s_val;
  }
};

/**
 * Binary operator class
 *
 * With the BinaryOp, we can write the +,-,*,/ for matrix.
 */
template<typename Op, typename LeftExpr, typename RightExpr, typename DataType>
class BinaryOp : public ExprBase<BinaryOp<Op, LeftExpr, RightExpr, DataType>,
    DataType> {
 public:
  const LeftExpr & left;
  const RightExpr & right;
  inline BinaryOp() {
  }
  ;
  inline BinaryOp(const LeftExpr & l, const RightExpr & r)
      : left(l),
        right(r) {
  }
  ;
  inline DataType eval(size_t i, size_t j) const {
    return Op::matrix_op(left.eval(i, j), right.eval(i, j));
  }

};

/**
 * Uninary operator class
 *
 * With the SingleOp, we can transform the matrix to another matrix.
 */
template<typename Op, typename Expr, typename DataType>
class SingleOp : public ExprBase<SingleOp<Op, Expr, DataType>, DataType> {
 public:
  const Expr & left;
  inline SingleOp() {
  }
  ;
  inline SingleOp(const Expr & l)
      : left(l) {
  }
  ;
  inline DataType eval(size_t i, size_t j) const {
    return Op::matrix_op(left.eval(i, j));
  }

};

}  //namespace matrix

} //namespace snoopy

#endif /* SNOOPY_MATRIX_EXPR_H */
