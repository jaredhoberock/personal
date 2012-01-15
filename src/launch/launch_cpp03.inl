#include "launch_cpp03.hpp"
#include "shared_storage_requirements_calculator.hpp"
#include <iostream>

template<typename Function>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f)
{
  std::cout << "launch: needs to dynamically allocate 0 bytes" << std::endl;
}

template<typename Function, typename Arg1>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5, arg6);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  shared_storage_requirements_calculator calc;

  static const std::size_t num_dynamic_smem_bytes = calc.calculate(f, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

