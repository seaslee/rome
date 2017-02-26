/**
 *  \file  allocator.h
 *  \author xuxinchao
 */

#ifndef SNOOPY_ALLOCATOR_H
#define SNOOPY_ALLOCATOR_H

#include <limits>
#include "../common/utils.h"

namespace snoopy {
namespace storage {

// -------------------------------
/// @Brief  Allocator is interface for allocate and deallocate memory
// ---------------------------------
class Allocator {
    public:
       virtual ~ Allocator() {}
       
       // -------------------------------
       /// @Brief  allocate `num_bytes` size memory with alignment of `align`
       /// 
       /// @Param align: the number that the return pointer shoule align
       /// @Param num_bytes: the number of the bytes to allocate
       /// 
       /// @Returns the pointer to the meomery
       // ---------------------------------
       virtual void * allocate_raw(size_t align, size_t num_bytes) = 0;
       
       // -------------------------------
       /// @Brief  deallocate the raw memory 
       /// 
       /// @Param ptr: the pointer of the memory to release
       // ---------------------------------
       virtual void deallocate_raw(void * ptr) = 0;

       // -------------------------------
       /// @Brief  allocate for the type `T`
       /// 
       /// @Param num_elements: the number of data to allocate
       /// 
       /// @Returns  the pointer to the memory
       // ---------------------------------
       template <typename T>
        inline T * allocate(size_t num_elements) {
           if (std::numeric_limits<size_t>::max() / sizeof(T) < num_elements) {
               return nullptr;
           }
           void * p = allocate_raw(default_align, sizeof(T) * num_elements);
           T * tp= reinterpret_cast<T*>(p);
           return tp;
        } 

       // -------------------------------
       /// @Brief  deallocate for the type `T`
       /// 
       /// @Param ptr: the pointer of the memory to release
       // ---------------------------------
        template <typename T>
        inline void deallocate(T * ptr) {
           if (ptr) {
               deallocate_raw(ptr);
           }
        }

    private:
        static const size_t default_align = 32;
};

class CPUallocator : public Allocator {
    public:
        CPUallocator() {}
        ~ CPUallocator() {}

    inline void * allocate_raw(size_t align, size_t num_bytes) {

        void * ptr = ptr;
        size_t required_align = sizeof(void *);
        if (num_bytes < required_align) {
            ptr = malloc(num_bytes);
        } else {
            if (posix_memalign(&ptr, align, num_bytes) != 0) {
                ptr = nullptr;
            }
        }

        return ptr;
    }

    inline void deallocate_raw(void * ptr) {
        free(ptr);
    }

    private:
        DISALLOW_COPY_AND_ASSIGN(CPUallocator)

};




}
}

#endif
