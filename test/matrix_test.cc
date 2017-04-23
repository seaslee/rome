#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h> 
#include "../matrix/matrix.h"

using namespace snoopy::matrix;

int main(int argc, char** argv) { 
    testing::InitGoogleTest(&argc, argv); 
    // Runs all tests using Google Test. 
    return RUN_ALL_TESTS(); 
}

struct Square {
  static double matrix_op(double x) {
    return (x * x);
  }
};

TEST(Matrix, ele_op_test) {
  Matrix<float, 3> m1 { { { 1, 2, 3 }, { 2, 3, 4 } },
      { { 1, 2, 3 }, { 2, 3, 4 } } };
  Matrix<float, 3> m2 { { { 1, 4, 3 }, { 6, 3, 4 } },
      { { 1, 4, 3 }, { 6, 3, 4 } } };
  EXPECT_EQ(m1.get_shape(), m2.get_shape());
  Matrix<float, 3> m3(m1.get_shape());
  //add
  m3 = m1 + m2;
  Matrix<float, 3> m4 = { { { 2, 6, 6 }, { 8, 6, 8 } }, { { 2, 6, 6 }, { 8, 6, 8 } } };
  EXPECT_EQ(m3, m4);
  //sub
  m3 = m1 - m2;
  m4 = { { {0, -2, 0}, {-4,0,0}}, { {0,-2,0}, {-4,0,0}}};
  EXPECT_EQ(m3, m4);
  //mul
  m3 = m1 * m2;
  m4 = { { {1, 8, 9}, {12,9,16}}, { {1,8,9}, {12,9,16}}};
  EXPECT_EQ(m3, m4);
  //div
  m3 = m1 / m2;
  m4 = { { {1, 0.5, 1}, {1.0/3,1,1}}, { {1,0.5,1}, {1.0/3,1,1}}};
  EXPECT_EQ(m3, m4);
  //scalar +
  m3 = m1 + 1;
  m4 = { { {2, 3, 4}, {3, 4, 5}}, { {2, 3, 4}, {3, 4, 5}}};
  EXPECT_EQ(m3, m4);
  //scalar -
  m3 = m1 - 1;
  m4 = { { {0, 1, 2}, {1, 2, 3}}, { {0, 1, 2}, {1, 2, 3}}};
  EXPECT_EQ(m3, m4);
  //scalar *
  m3 = m1 * 2;
  m4 = { { {2, 4, 6}, {4, 6, 8}}, { {2, 4, 6}, {4, 6, 8}}};
  EXPECT_EQ(m3, m4);
  //scalar /
  m3 = m4 / 2;
  EXPECT_EQ(m3, m1);

  m1 += 1;
  m4 = { { {2, 3, 4}, {3, 4, 5}}, { {2, 3, 4}, {3, 4, 5}}};
  EXPECT_EQ(m1, m4);
  m1 -= 1;
  m4 = { { { 1, 2, 3 }, { 2, 3, 4 } },
      { { 1, 2, 3 }, { 2, 3, 4 } } };
  EXPECT_EQ(m1, m4);
  m1 *= 2;
  m4 = { { {2, 4, 6}, {4, 6, 8}}, { {2, 4, 6}, {4, 6, 8}}};
  EXPECT_EQ(m1, m4);
  m1 /= 2;
  m4 = { { { 1, 2, 3 }, { 2, 3, 4 } },
      { { 1, 2, 3 }, { 2, 3, 4 } } };
  EXPECT_EQ(m1, m4);
  //add
  m1 += m2;
  m4 = { { { 2, 6, 6 }, { 8, 6, 8 } }, { { 2, 6, 6 }, { 8, 6, 8 } } };
  EXPECT_EQ(m1, m4);


}

TEST(Matrix, dot_test) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 2> m2 { { 2, 3 }, { 2, 3 }, {2, 3} };

  Matrix<float, 2> m3 = dot(m1, m2, 1.f, false);
  Matrix<float, 2> m4 = dot(m1, m2);
  Matrix<float, 2> m5 {{12, 18}, {18, 27}};

  EXPECT_EQ(m3, m5);
  EXPECT_EQ(m4, m5);

  Matrix<float, 2> slice1 {{1,2,3}};
  Matrix<float, 2> s = m1.slice(0, 1);
  EXPECT_EQ(slice1, s);
}

TEST(Matrix, matrix_sum_test) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 1> m2 = sum(m1, 0);
  Matrix<float, 1> temp { 3, 5, 7 };
  EXPECT_EQ(m2, temp);
  Matrix<float, 1> m3 = sum(m1, 1);
  Matrix<float, 1> temp1 { 6, 9 };
  EXPECT_EQ(m3, temp1);
  Matrix<float, 1> m4 = sum(m1, 2);
  Matrix<float, 1> temp2 { 15 };
  EXPECT_EQ(m4, temp2);
  cerr << "step 1" << endl;

  MatrixShape<2> s{1,3};
  Matrix<float, 2> m5(s);
  Matrix<float, 2> m6 {{3, 5, 7}};
  sum(m5, m1, 0);
  EXPECT_EQ(m5, m6);

  MatrixShape<2> s1{2,1};
  Matrix<float, 2> m7(s1);
  Matrix<float, 2> m8 {{6}, {9}};
  sum(m7, m1, 1);
  EXPECT_EQ(m7, m8);

  repmat(m1, m6, 0);
  Matrix<float, 2> m9 {{3, 5, 7}, {3, 5, 7}};
  EXPECT_EQ(m1, m9);

  repmat(m1, m8, 1);
  Matrix<float, 2> m10 {{6, 6, 6}, {9, 9, 9}};
  EXPECT_EQ(m1, m10);

}

TEST(Matrix, matrix_max_test) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  vector<int> max_index_vector(2,0);
  Matrix<float, 1> m2 = max(m1, max_index_vector, 0);
  Matrix<float, 1> temp { 2, 3, 4 };
  EXPECT_EQ(m2, temp);

  Matrix<float, 1> m3 = max(m1, max_index_vector, 1);
  Matrix<float, 1> temp1 {  3, 4 };
  EXPECT_EQ(m3, temp1);

  Matrix<float, 1> m4 = max(m1, max_index_vector, 2);
  Matrix<float, 1> temp2 { 4 };
  EXPECT_EQ(m4, temp2);

}

TEST(Matrix, copy_from) {
  Matrix<float, 2> m1 { { 1, 2, 3 }, { 2, 3, 4 } };
  Matrix<float, 2> m2 { { 4, 5, 3 }, { 2, 3, 4 }, {5, 6, 6}};
  Matrix<float, 2> m3 { { 4, 5, 3 }, { 5, 6, 6 } };

  m1[0].copy_from(m2[0]);
  m1[1].copy_from(m2[2]);

  EXPECT_EQ(m1, m3);
}

TEST(Matrix, test_softmax) {
  Matrix<float, 2> m1 {{1, 2, 3}, {1, 3, 5}, {1, 2, 2}, {2, 3, 3}};
  MatrixShape<2> s1{4, 3};
  Matrix<float, 2> m2(s1);
  softmax(m2, m1);
  Matrix<float, 2> m3 
      {{ 0.09003057,  0.24472847,  0.66524096},
      { 0.01587624,  0.11731043,  0.86681333},
      { 0.1553624,   0.4223188,   0.4223188 },
      { 0.1553624,   0.4223188,   0.4223188 }};
  EXPECT_EQ(m2, m3);
}


