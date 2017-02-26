#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete;


// check condition is satisified
// assert with condition and output the error string
inline void check(int condition, string s) {
  if (!condition) {
    cerr << s << endl;
    exit(-1);
  }
}

inline void check_shape(int condition, string s) {
  if (!condition) {
    cerr << s << endl;
    exit(-1);
  }
}

#endif //MATRIX_UTILS_H_
