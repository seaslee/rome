/**
 *  \file  buffer.h
 *  \author xuxinchao
 */

#ifndef SNOOPY_BUFFER_H
#define SNOOPY_BUFFER_H

#include <atomic>
#include "allocator.h"
#include "../common/utils.h"

namespace snoopy {
namespace storage {

class RefCount {
    public:
        inline RefCount():_ref_count(1) {}

        // -------------------------------
        /// @Brief  increase the ref count by 1
        // ---------------------------------
        inline void ref() const {
            _ref_count.fetch_add(1, std::memory_order_relaxed);
        }

        // -------------------------------
        /// @Brief  decrease the ref count by 1
        /// 
        /// @Returns   true if ref count is zero, false otherwise
        //
        // ---------------------------------
        inline bool unref() const {
            if (_ref_count.load(memory_order_acquire) == 1 ||
                    _ref_count.fetch_sub(1) == 1) {
                _ref_count.store(0);
                delete this;
                return true;
            } else {
                return false;
            }
        }

        inline bool ref_is_one() const {
            return (_ref_count.load(std::memory_order_acquire) == 1);
        }

    protected:
        virtual ~RefCount() {}

    private:
        mutable std::atomic_int_fast32_t _ref_count;

        DISALLOW_COPY_AND_ASSIGN(RefCount)
        
};

template <typename T>
class TensorBuffer : public RefCount {
    public:
        virtual T * data() const = 0;
        virtual T * data()  = 0;
        virtual int64_t size() const = 0; 
        virtual TensorBuffer * root_buffer() = 0;
        virtual const T & operator[] (size_t i) const = 0;
        virtual T & operator[] (size_t i) = 0;
        virtual const T & at (size_t i) const = 0;
        virtual T & at (size_t i) = 0;
        ~ TensorBuffer() {}
};

template <typename T>
class Buffer : public TensorBuffer<T> {
    public:
        Buffer(Allocator * a, int64_t n) : _alloc(a), _data(a->allocate<T>(n)), _data_ele_num(n) {}
        T * data() const {return _data; }
        T * data() { return _data; }
        int64_t size() const { return _data_ele_num * sizeof(T); }
        TensorBuffer<T> * root_buffer() {return this;}
        virtual const T & operator[] (size_t i) const { return (_data[i]); }
        virtual T & operator[] (size_t i) { return (_data[i]); }
        virtual const T & at (size_t i) const { return (_data[i]); }
        virtual T & at (size_t i) { return (_data[i]); }
    private:
        inline ~Buffer() { 
            _alloc->deallocate(_data);
        };
    private:
        Allocator * _alloc;
        int64_t _data_ele_num;
        T * _data;
};

template <typename T>
class SubBuffer : public TensorBuffer<T> {
    public:
        SubBuffer(TensorBuffer<T> * tb, size_t offset):
            _root(tb),
            _data(reinterpret_cast<T *> (tb->data() + offset)) { 
            //not forget to ref
            _root->ref();
            }

        T * data() const { return _data; }
        T * data() { return _data; }
        int64_t size() const { return _data_ele_num * sizeof(T); }
        TensorBuffer<T> * root_buffer() {return _root;}
        virtual const T & operator[] (size_t i) const { return _data[i]; }
        virtual T & operator[] (size_t i) { return _data[i]; }
        virtual const T & at (size_t i) const { return (_data[i]); }
        virtual T & at (size_t i) {return (_data[i]); }

    private:
        ~SubBuffer() { _root->unref(); };
    private:
        TensorBuffer<T> * _root;
        int64_t _data_ele_num;
        T * _data;
};


}
}

#endif
