#pragma once

#include <memory>

namespace detail
{

template<unsigned int N, bool terminate = N <= sizeof(char)> struct static_storage_impl;

template<unsigned int N>
  struct static_storage_impl<N, true>
{
  char impl;

  __device__ inline void *void_ptr()
  {
    return reinterpret_cast<void*>(&impl);
  }

  __device__ inline const void *void_ptr() const
  {
    return reinterpret_cast<const void*>(&impl);
  }
};

template<unsigned int N>
  struct static_storage_impl<N,false>
    : static_storage_impl<N - sizeof(char)>
{
  char impl;
};

// static_storage is a type with size at least N bytes
template<unsigned int N>
  struct static_storage
    : static_storage_impl<N>
{};

} // end detail

template<typename T>
  class uninitialized
    : private detail::static_storage<sizeof(T)>
{
  private:
    detail::static_storage<sizeof(T)> impl;

    typedef detail::static_storage<sizeof(T)> super_t;

    __device__ inline const T* ptr() const
    {
      return reinterpret_cast<const T*>(super_t::void_ptr());
    }

    __device__ inline T* ptr()
    {
      return reinterpret_cast<T*>(super_t::void_ptr());
    }

  public:
    // copy assignment
    __device__ inline uninitialized<T> &operator=(const T &other)
    {
      T& self = *this;
      self = other;
      return *this;
    }

    __device__ inline T& get()
    {
      return *ptr();
    }

    __device__ inline const T& get() const
    {
      return *ptr();
    }

    __device__ inline operator T& ()
    {
      return get();
    }

    __device__ inline operator const T&() const
    {
      return get();
    }

    inline __device__ void construct()
    {
      ::new(ptr()) T();
    }

    template<typename Arg>
    inline __device__ void construct(const Arg &a)
    {
      ::new(ptr()) T(a);
    }

    template<typename Arg1, typename Arg2>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2)
    {
      ::new(ptr()) T(a1,a2);
    }

    template<typename Arg1, typename Arg2, typename Arg3>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3)
    {
      ::new(ptr()) T(a1,a2,a3);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4)
    {
      ::new(ptr()) T(a1,a2,a3,a4);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8,a9);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
    inline __device__ void construct(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9, const Arg10 &a10)
    {
      ::new(ptr()) T(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);
    }

    inline __device__ void destroy()
    {
      T& self = *this;
      self.~T();
    }

    template<typename Arg1>
    inline __device__ void destroy(const Arg1 &a1)
    {
      T& self = *this;
      self.~T(a1);
    }

    template<typename Arg1, typename Arg2>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2)
    {
      T& self = *this;
      self.~T(a1,a2);
    }

    template<typename Arg1, typename Arg2, typename Arg3>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3)
    {
      T& self = *this;
      self.~T(a1,a2,a3);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5,a6);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5,a6,a7);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5,a6,a7,a8);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5,a6,a7,a8,a9);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
    inline __device__ void destroy(const Arg1 &a1, const Arg2 &a2, const Arg3 &a3, const Arg4 &a4, const Arg5 &a5, const Arg6 &a6, const Arg7 &a7, const Arg8 &a8, const Arg9 &a9, const Arg10 &a10)
    {
      T& self = *this;
      self.~T(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);
    }
};

