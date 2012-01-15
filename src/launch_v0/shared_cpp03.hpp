#pragma once

#include <utility>

template<typename T>
  class shared
{
  private:
    shared(const shared &);
    shared();

    template<typename> friend class launch_core_access;

    // allows friends to construct dummy shared<T> objects
    struct construct_dummy_tag {};
    shared(construct_dummy_tag) {}

    shared(void *p)
      : ptr(reinterpret_cast<T*>(p))
    {
      ::new(static_cast<void*>(ptr)) T();
    }

    template<typename Arg1>
      shared(void *p, const Arg1 &arg1)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1);
    }

    template<typename Arg1, typename Arg2>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2);
    }

    template<typename Arg1, typename Arg2, typename Arg3>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5,arg6);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
    }

    template<typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
      shared(void *p, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9, const Arg10 &arg10)
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10);
    }

    ~shared()
    {
      // destroy the object
      ptr->~T();
    }

    T *ptr;
};

template<typename T>
  struct launch_core_access
{
  static shared<T> &shared_dummy()
  {
    static shared<T> dummy((typename shared<T>::construct_dummy_tag()));
    return dummy;
  }
};

