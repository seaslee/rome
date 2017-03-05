#ifndef MATRIX_UTILS_H_
#define MATRIX_UTILS_H_
#include <iostream>
#include <string>
#include <vector>
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

//
inline void split(const std::string & str, std::vector<std::string> & v, 
            const std::string & delim) {
    size_t pre_pos = 0;
	size_t position = 0;
    std::string tmp_str;
    v.clear();
    if (str.empty()) {
        return;
    }
    while ((position = str.find(delim, pre_pos)) != std::string::npos) {
        tmp_str.assign(str, pre_pos, position - pre_pos);
        if (!tmp_str.empty()) {
            v.push_back(tmp_str);
        }
        pre_pos = position + delim.length();
    }
    tmp_str.assign(str, pre_pos, str.length() - pre_pos);
    if (!tmp_str.empty()) {
        v.push_back(tmp_str);
    }	
}


#endif //MATRIX_UTILS_H_
