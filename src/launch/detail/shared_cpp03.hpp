#pragma once

#include <utility>
#include <thrust/tuple.h>
#include "marshaled_args.hpp"
#include "tuple_placement_new.hpp"
#include "../this_thread_group.hpp"

template<typename T>
  class shared
{
  public:
    __host__ __device__
    T &get() const
    {
      return *ptr;
    }

  private:
    shared(const shared &);
    shared();

    friend class launch_core_access;

    // allows friends to construct dummy shared<T> objects
    struct construct_dummy_tag {};
    shared(construct_dummy_tag) {}

    __device__
    shared(void *p)
      : ptr(reinterpret_cast<T*>(p))
    {
      if(threadIdx.x == 0)
      {
        ::new(static_cast<void*>(ptr)) T();
      } // end if

      __syncthreads();
    }

    template<typename Arg>
    __device__
      shared(void *p, const Arg &arg)
        : ptr(reinterpret_cast<T*>(p))
    {
      if(this_thread_group::get_thread_id() == 0)
      {
        // construct the object given the arg
        ::new(static_cast<void*>(ptr)) T(arg);
      } // end if

      __syncthreads();
    }

    template<typename Tuple>
    __device__
      shared(const detail::marshaled_args<Tuple> &ma)
        : ptr(reinterpret_cast<T*>(ma.ptr()))
    {
      if(this_thread_group::get_thread_id() == 0)
      {
        // construct the object given the args
        tuple_placement_new<T>(static_cast<void*>(ptr), ma.args());
      } // end if

      __syncthreads();
    }

    __host__ __device__
    ~shared()
    {
#if __CUDA_ARCH__
      __syncthreads();

      if(this_thread_group::get_thread_id() == 0)
      {
        // destroy the object
        ptr->~T();
      } // end if
#endif // __CUDA_ARCH__
    }

    T *ptr;
};

