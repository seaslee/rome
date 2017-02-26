#include <cstdlib>
#include "allocator.h"

namespace snoopy {
namespace storage {

template <typename T>
T * Allocator::allocate(size_t num_elements) {
   if (std::numeric_limits<size_t>::max() / sizeof(T) < num_elements) {
       return nullptr;
   }
   void * p = allocate_raw(default_align, sizeof(T) * num_elements);
   T * tp= reinterpret_cast<T*>(p);
   return tp;
} 


template <typename T>
void Allocator::deallocate(T * ptr) {
   if (ptr) {
       deallocate_raw(ptr);
   }
}

void * CPUallocator::allocate_raw(size_t align, size_t num_bytes) {

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


void CPUallocator::deallocate_raw(void * ptr) {
    free(ptr);
}

} //namespace storage 
} //namespace snoopy
