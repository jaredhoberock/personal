#pragma once

#include <cstddef>
#include "../shared.hpp"
#include "launch_core_access.hpp"
#include <vector>

class shared_storage_requirements_calculator
{
  public:
    typedef std::vector<std::size_t> result_type;

    shared_storage_requirements_calculator()
      : m_result()
    {}

    template<typename Arg>
      struct arg_converter
    {
      arg_converter(shared_storage_requirements_calculator &calc, const Arg &arg)
        : m_calc(calc),m_arg(arg)
      {}

      operator const Arg& () const
      {
        m_calc.encountered_thread_local<Arg>();

        return m_arg;
      }

      template<typename T>
        operator const shared<T> & ()
      {
        m_calc.encountered_shared<T>();

        return launch_core_access::shared_dummy<T>();
      }

      template<typename T>
        operator shared<T> & ()
      {
        m_calc.encountered_shared<T>();

        return launch_core_access::shared_dummy<T>();
      }

      shared_storage_requirements_calculator &m_calc;
      const Arg& m_arg;
    };

    template<typename T>
      static arg_converter<T> make_arg_converter(shared_storage_requirements_calculator &calc, const T &arg)
    {
      return arg_converter<T>(calc,arg);
    }

#if defined(__GCC_EXPERIMENTAL_CXX0X__)
    template<typename Function, typename... Args>
      result_type calculate(Function f, Args&&... args)
    {
      reset(sizeof...(Args));

      // convert to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this, args)...);
      }
      catch(...) {}

      return m_result;
    }
#else
    template<typename Function, typename Arg1>
      result_type calculate(Function f, const Arg1 &arg1)
    {
      reset(1);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2)
    {
      reset(2);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3)
    {
      reset(3);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4)
    {
      reset(4);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5)
    {
      reset(5);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4), make_arg_converter(*this,arg5));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6)
    {
      reset(6);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4), make_arg_converter(*this,arg5), make_arg_converter(*this,arg6));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7)
    {
      reset(7);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1),
          make_arg_converter(*this,arg2),
          make_arg_converter(*this,arg3),
          make_arg_converter(*this,arg4),
          make_arg_converter(*this,arg5),
          make_arg_converter(*this,arg6),
          make_arg_converter(*this,arg7));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8)
    {
      reset(8);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1),
          make_arg_converter(*this,arg2),
          make_arg_converter(*this,arg3),
          make_arg_converter(*this,arg4),
          make_arg_converter(*this,arg5),
          make_arg_converter(*this,arg6),
          make_arg_converter(*this,arg7),
          make_arg_converter(*this,arg8));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9)
    {
      reset(9);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1),
          make_arg_converter(*this,arg2),
          make_arg_converter(*this,arg3),
          make_arg_converter(*this,arg4),
          make_arg_converter(*this,arg5),
          make_arg_converter(*this,arg6),
          make_arg_converter(*this,arg7),
          make_arg_converter(*this,arg8),
          make_arg_converter(*this,arg9));
      }
      catch(...) {}

      return m_result;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
      result_type calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9, const Arg10 &arg10)
    {
      reset(10);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1),
          make_arg_converter(*this,arg2),
          make_arg_converter(*this,arg3),
          make_arg_converter(*this,arg4),
          make_arg_converter(*this,arg5),
          make_arg_converter(*this,arg6),
          make_arg_converter(*this,arg7),
          make_arg_converter(*this,arg8),
          make_arg_converter(*this,arg9),
          make_arg_converter(*this,arg10));
      }
      catch(...) {}

      return m_result;
    }
#endif

  private:
    void reset(std::size_t num_args)
    {
      m_result.clear();
      m_result.reserve(num_args);
    }

    // when encountering shared<T>, accumulate
    template<typename T>
      void encountered_shared()
    {
      m_result.push_back(sizeof(T));

      // bail out if we're done converting
      if(m_result.size() == m_result.capacity())
      {
        throw 13;
      }
    }

    template<typename T>
      void encountered_thread_local()
    {
      m_result.push_back(0);

      // bail out if we're done converting
      if(m_result.size() == m_result.capacity())
      {
        throw 13;
      }
    }

    result_type m_result;
};

