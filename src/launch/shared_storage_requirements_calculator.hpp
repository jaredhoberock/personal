#pragma once

#include <cstddef>
#include "shared.hpp"

class shared_storage_requirements_calculator
{
  public:
    shared_storage_requirements_calculator()
      : m_total_num_arguments(),
        m_num_arguments_inspected(),
        m_shared_storage_size()
    {}

#if defined(__GCC_EXPERIMENTAL_CXX0X__)
    template<typename Function, typename... Args>
      std::size_t calculate(Function f, Args&&...)
    {
      reset(sizeof...(Args));

      // convert this to parameters but avoid calling the function
      try
      {
        f(self<Args>()...);
      }
      catch(...) {}

      return m_shared_storage_size;
    }
#else
    template<typename Function, typename Arg1>
      std::size_t calculate(Function f, const Arg1 &)
    {
      reset(1);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &)
    {
      reset(2);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &)
    {
      reset(3);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &)
    {
      reset(4);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &)
    {
      reset(5);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &, const Arg6 &)
    {
      reset(6);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &, const Arg6 &, const Arg7 &)
    {
      reset(7);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &, const Arg6 &, const Arg7 &, const Arg8 &)
    {
      reset(8);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &, const Arg6 &, const Arg7 &, const Arg8 &, const Arg9 &)
    {
      reset(9);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
      std::size_t calculate(Function f, const Arg1 &, const Arg2 &, const Arg3 &, const Arg4 &, const Arg5 &, const Arg6 &, const Arg7 &, const Arg8 &, const Arg9 &, const Arg10 &)
    {
      reset(10);

      // convert this to parameters but avoid calling the function
      try
      {
        f(*this, *this, *this, *this, *this, *this, *this, *this, *this, *this);
      }
      catch(...) {}

      return m_shared_storage_size;
    }
#endif

  private:
    template<typename T>
      shared_storage_requirements_calculator &self()
    {
      return *this;
    }

    void reset(std::size_t num_args)
    {
      m_total_num_arguments = num_args;
      m_num_arguments_inspected = 0;
      m_shared_storage_size = 0;
    }

    // convert to anything
    template<typename T> inline operator T& () const {}
    template<typename T> inline operator const T& () const{}

    // when converting to shared<T>, accumulate
    template<typename T>
      operator shared<T> & ()
    {
      m_shared_storage_size += sizeof(T);
      ++m_num_arguments_inspected;

      // bail out if we're done converting
      if(m_num_arguments_inspected == m_total_num_arguments)
      {
        throw 13;
      }

      return launch_core_access<T>::shared_dummy();
    }

    template<typename T>
      operator const shared<T> & ()
    {
      m_shared_storage_size += sizeof(T);
      ++m_num_arguments_inspected;

      // bail out if we're done converting
      if(m_num_arguments_inspected == m_total_num_arguments)
      {
        throw 13;
      }

      return launch_core_access<T>::shared_dummy();
    }

    std::size_t m_total_num_arguments;
    std::size_t m_num_arguments_inspected;
    std::size_t m_shared_storage_size;
};

