/**
 *  Adapter class for fixed dimension matrix in order to not care the 
 *  dimension of the data in the neural network implementation.
 *
 */

#ifndef SNOOPY_MATRIX_MATRIX_BLOB_H_
#define SNOOPY_MATRIX_MATRIX_BLOB_H_

#include "matrix.h"

namespace snoopy {

namespace matrix {
    
    struct BlobShape {

      BlobShape()
           :dims_(0),
            size_(0),
            start_(0), 
            shape_(nullptr)
            {
            };

      BlobShape(const BlobShape &s):
       dims_(s.dims_),
       size_(s.size_),
       start_(s.start_),
       shape_(new size_t[dims_])
       {
           std::copy(s.shape_, s.shape_ + s.dims_, shape_); 
       }

      template <size_t N>
      BlobShape(const MatrixShape<N> mat_shape):
      dims_(N),
      size_(mat_shape.size),
      start_(mat_shape.start),
      shape_(new size_t[dims_]) {
          std::copy(mat_shape.shape, mat_shape.shape + N, shape_);
      }

      BlobShape(std::initializer_list<size_t> sp):
       dims_(sp.size()),
       start_(0)
       {
           shape_ = new size_t[dims_];
           int i = 0;
           for (auto x : sp) {
               size_ *= x;
               shape_[i] = x;
               ++i;
           }
       }

      BlobShape(const std::vector<size_t> & sp):
          dims_(sp.size()),
          start_(0) {
          shape_ = new size_t[dims_];
          for(int i = 0; i < sp.size(); ++i) {
              size_ *= sp[i];
              shape_[i] = sp[i];
          }
      }

      BlobShape(const BlobShapeProto & sp):
          dims_(sp.dim_size()),
          start_(0) {
          shape_ = new size_t[dims_];
          for(int i = 0; i < sp.dim_size(); ++i) {
              size_ *= sp.dim(i);
              shape_[i] = sp.dim(i);
          }
      }


      BlobShape & operator =(const BlobShape &s) {
          //check the shape 
          if (&s != this) {
              if (dims_ < s.dims_) {
                  delete shape_;
                  dims_ = s.dims_;
                  shape_ = new size_t[dims_];
                  std::copy(s.shape_, s.shape_ + s.dims_, shape_); 
              } else {
                  dims_ = s.dims_;
                  std::copy(s.shape_, s.shape_ + s.dims_, shape_); 
              }
              size_ = s.size_;
              start_ = s.start_;
          }
          return *this;
      }

      bool operator ==(const BlobShape &s) {
          if (s.dims_ != dims_) {
            return false;
          }
          for (size_t bs_index = 0; bs_index < dims_; ++bs_index) {
            if (shape_[bs_index] != s.shape_[bs_index]) {
                return false;
            }
          }
          return true; 
      }

      MatrixShape<2> flatten_2d() {
          CHECK_GE(dims_, 2);
          size_t dim1 = 1;
          size_t dim2 = shape_[dims_-1];
          for (int i = 0; i < dims_ - 1; ++i) {
            dim1 *= shape_[i];
          }
          MatrixShape<2> tmp_shape {dim1, dim2};
          return tmp_shape;
      }

      size_t & operator [](size_t i) {
        return shape_[i];
      }

      const size_t & operator [](size_t i) const {
        return shape_[i];
      }

      size_t dims() {
        return dims_;
      }

      const size_t dims() const {
        return dims_;
      }

      size_t get_size() { return size_; }
      const size_t get_size() const { return size_; }

      const size_t get_length() const {
        size_t len = 1;
         for (int i = 0; i < dims_; ++i) {
            len *= shape_[i]; 
         } 
         return len;
      }

      int dims_;
      size_t size_; ///< the total size of all dimension
      size_t start_; ///< the start offset
      size_t * shape_; ///< store size of each dimension
    }; 


    template <typename DataType>
    struct MBlob {
       BlobShape blob_shape_;
       storage::TensorBuffer<DataType> * data_; 
       size_t stride_;

       MBlob():data_(nullptr), stride_(0) {}

       MBlob(storage::TensorBuffer<DataType> * data, 
            const BlobShape & bs):
           blob_shape_(bs), 
           stride_(bs.shape_[bs.dims()-1]),
           data_(data) {    
           data->ref(); // NOTE!!
        }
        
       template <size_t N>
       MBlob(const Matrix<DataType, N> & mat, const BlobShape & bs) {
           data_ = mat.get_data();
           data_->ref();
           blob_shape_ = bs;
           stride_ = mat.get_stride();
       }
       
       MBlob(storage::Allocator * a,
               const BlobShape & bs) {
           blob_shape_ = bs;
           size_t len = blob_shape_.get_length();
           data_ = new storage::Buffer<DataType>(a, len);
           stride_ = bs[bs.dims()-1];
       }

       inline Matrix<DataType, 2> flatten_2d_matrix() {
          return Matrix<DataType, 2>(data_, blob_shape_.flatten_2d(), stride_);
       }

       size_t dim_at(size_t index) {
            return blob_shape_[index];
       }

       DataType at(size_t index) {
           return data_->at(index);
       }
    };

    template <typename DataType> 
    struct Blob {
       shared_ptr<MBlob<DataType> > data_;
       shared_ptr<MBlob <DataType> > diff_; 

       Blob(MBlob<DataType> * data, MBlob<DataType> * diff):
           data_(data), diff_(diff) {}

       Blob(MBlob<DataType> * data) :
           data_(data),
           diff_(nullptr) {}

       shared_ptr<MBlob<DataType> > get_data() {return data_;}
       shared_ptr<MBlob<DataType> > get_diff() {return diff_;}

       void set_data(MBlob<DataType> * mb) {data_ = shared_ptr<MBlob<DataType> >(mb);}
       void set_diff(MBlob<DataType> * mb) {diff_ = shared_ptr<MBlob<DataType> > (mb);}

       void set_data(shared_ptr<MBlob<DataType> > mb) {data_ = mb;}
       void set_diff(shared_ptr<MBlob<DataType> > mb) {diff_ = mb;}

       BlobShape get_blobshape() {return data_->blob_shape_;}
        
       //read protobuf
       int from_proto(const BlobParameter & bp) {
           return 0; 
       } 
       //write protobuf
       int to_proto(BlobParameter * bp, bool is_write_diff) {
           return 0; 
       } 

       inline size_t dim_at(size_t index) {
           return data_->dim_at(index);
       }

       inline void set_data_at(size_t index, const DataType & v) {
           data_->data_->at(index) = v;
       }

       inline void set_diff_at(size_t index, const DataType & v) {
           diff_->data_->at(index) = v;
       }

       inline DataType get_data_at(size_t index) {
           return data_->data_->at(index);
       }

       inline DataType get_diff_at(size_t index) {
           return diff_->data_->at(index);
       }

       inline size_t get_count() {
           return data_->blob_shape_.get_size();
       }

    };

    //helper function to create Blob object
    template<typename DataType>
    shared_ptr<Blob<DataType> > create_blob_object(const BlobShape & bs, bool is_create_diff) {    
        storage::Allocator * data_allocator = new storage::CPUallocator;
        size_t capicity = bs.get_length();
        data_allocator->allocate<DataType>(capicity);
        MBlob<DataType> * data_blob = new MBlob<DataType>(data_allocator, bs);
        MBlob<DataType> * diff_blob = nullptr;

        if (is_create_diff) {
            storage::Allocator * diff_allocator = new storage::CPUallocator;
            diff_allocator->allocate<DataType>(capicity);
            diff_blob = new MBlob<DataType>(diff_allocator, bs);
        }

        shared_ptr<Blob<DataType> > blob (new Blob<DataType>(data_blob, diff_blob));
        return blob;
    }

    //helper function to create Blob object from BlobParameter
    template<typename DataType>
    shared_ptr<Blob<DataType> > create_blob_object(const BlobParameter & bp) {
        vector<size_t> shape_vector;
        for (int i = 0; i < bp.shape().dim_size(); ++i) {
            shape_vector.push_back(bp.shape().dim(i));
        }
        BlobShape bs(shape_vector);
        shared_ptr<Blob<DataType> > blob_ptr;
        if (bp.diff_size() > 0) {
            blob_ptr =  create_blob_object<DataType>(bs, true);
            if (bp.data_size() > 0) {
                //load
                CHECK_EQ(blob_ptr->get_count(), bp.data_size());
                CHECK_EQ(bp.diff_size(), bp.data_size());
                for (int i = 0; i < bp.data_size(); ++ i){
                    blob_ptr->set_data_at(i, bp.data(i));
                    blob_ptr->set_diff_at(i, bp.diff(i));
                }
            } 
        } else {
            blob_ptr =  create_blob_object<DataType>(bs, false);
            if (bp.data_size() > 0) {
                //load
                CHECK_EQ(blob_ptr->get_count(), bp.data_size());
                for (int i = 0; i < bp.data_size(); ++ i){
                    blob_ptr->set_data_at(i, bp.data(i));
                }
            } 
        }
        return blob_ptr;
    } 

    template<typename DataType>
    shared_ptr<Blob<DataType> > create_blob_object(const BlobShapeProto & bsp, bool is_create_diff) {   
        BlobShape bs(bsp);
        create_blob_object<DataType>(bs, is_create_diff);
    } 


}

} //end namspace snoopy

#endif
