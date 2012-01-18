#pragma once

#include <cstddef>
#include "../shared.hpp"
#include "launch_core_access.hpp"

class shared_storage_requirements_calculator
{
  public:
    shared_storage_requirements_calculator()
      : m_total_num_arguments(),
        m_num_arguments_inspected(),
        m_shared_storage_size()
    {}

    template<typename Arg>
      struct arg_converter
    {
      arg_converter(shared_storage_requirements_calculator &calc, const Arg &arg)
        : m_calc(calc),m_arg(arg)
      {}

      operator const Arg& () const
      {
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
      std::size_t calculate(Function f, Args&&... args)
    {
      reset(sizeof...(Args));

      // convert to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this, args)...);
      }
      catch(...) {}

      return m_shared_storage_size;
    }
#else
    template<typename Function, typename Arg1>
      std::size_t calculate(Function f, const Arg1 &arg1)
    {
      reset(1);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2)
    {
      reset(2);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3)
    {
      reset(3);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4)
    {
      reset(4);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5)
    {
      reset(5);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4), make_arg_converter(*this,arg5));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6)
    {
      reset(6);

      // convert this to parameters but avoid calling the function
      try
      {
        f(make_arg_converter(*this,arg1), make_arg_converter(*this,arg2), make_arg_converter(*this,arg3), make_arg_converter(*this,arg4), make_arg_converter(*this,arg5), make_arg_converter(*this,arg6));
      }
      catch(...) {}

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7)
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

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8)
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

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9)
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

      return m_shared_storage_size;
    }

    template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
      std::size_t calculate(Function f, const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5, const Arg6 &arg6, const Arg7 &arg7, const Arg8 &arg8, const Arg9 &arg9, const Arg10 &arg10)
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

      return m_shared_storage_size;
    }
#endif

  private:
    void reset(std::size_t num_args)
    {
      m_total_num_arguments = num_args;
      m_num_arguments_inspected = 0;
      m_shared_storage_size = 0;
    }

    // when encountering shared<T>, accumulate
    template<typename T>
      void encountered_shared()
    {
      m_shared_storage_size += sizeof(T);
      ++m_num_arguments_inspected;

      // bail out if we're done converting
      if(m_num_arguments_inspected == m_total_num_arguments)
      {
        std::cout << "bailing out" << std::endl;
        throw 13;
      }
    }

    std::size_t m_total_num_arguments;
    std::size_t m_num_arguments_inspected;
    std::size_t m_shared_storage_size;
};

