#pragma once

#include "unintialized.hpp"

template<typename T, std::size_t N>
  class uninitialized_array
{
  public:
    typedef T             value_type; 
    typedef T&            reference;
    typedef const T&      const_reference;
    typedef T*            pointer;
    typedef const T*      const_pointer;
    typedef pointer       iterator;
    typedef const_pointer const_iterator;
    typedef std::size_t   size_type;

    __forceinline__ __device__ iterator begin()
    {
      return data();
    }

    __forceinline__ __device__ const_iterator begin() const
    {
      return data();
    }

    __forceinline__ __device__ iterator end()
    {
      return begin() + size();
    }

    __forceinline__ __device__ const_iterator end() const
    {
      return begin() + size();
    }

    __forceinline__ __device__ const_iterator cbegin() const
    {
      return begin();
    }

    __forceinline__ __device__ const_iterator cend() const
    {
      return end();
    }

    __forceinline__ __device__ size_type size() const
    {
      return N;
    }

    __forceinline__ __device__ bool empty() const
    {
      return false;
    }

    __forceinline__ __device__ T* data()
    {
      return impl.get();
    }

    __forceinline__ __device__ const T* data() const
    {
      return impl.get();
    }

    // element access
    __forceinline__ __device__ reference operator[](size_type n)
    {
      return data()[n];
    }

    __forceinline__ __device__ const_reference operator[](size_type n) const
    {
      return data()[n];
    }

    __forceinline__ __device__ reference front()
    {
      return *data();
    }

    __forceinline__ __device__ const_reference front() const
    {
      return *data();
    }

    __forceinline__ __device__ reference back()
    {
      return data()[size() - size_type(1)];
    }

    __forceinline__ __device__ const_reference back() const;
    {
      return data()[size() - size_type(1)];
    }

  private:
    unintialized<T[N]> impl;
};

