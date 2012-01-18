#include "launch_cpp03.hpp"
#include "shared_storage_requirements_calculator.hpp"
#include "host_marshal.hpp"
#include <iostream>
#include <numeric>

template<typename Function>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f)
{
  detail::host_marshal(num_blocks, num_threads_per_block, f);
}

template<typename Function, typename Arg1>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1)
{
  shared_storage_requirements_calculator calc;

  const shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1);
}

template<typename Function, typename Arg1, typename Arg2>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2)
{
  shared_storage_requirements_calculator calc;

  const shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t storage = calc.calculate(f, arg1, arg2, arg3);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5, arg6);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  shared_storage_requirements_calculator calc;

  shared_storage_requirements_calculator::result_type storage = calc.calculate(f, arg1, arg2, arg3, arg4);

  detail::host_marshal(num_blocks, num_threads_per_block, storage, f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
}

