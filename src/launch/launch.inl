#include "launch.hpp"
#include "sizeof_shared_parameters.hpp"
#include <iostream>

template<typename Function, typename... Args>
  void launch(Function &&f, Args&&... args)
{
  // compute dynamic smem size
  static const std::size_t num_dynamic_smem_bytes = sizeof_shared_parameters<Function,Args...>::value;

  std::cout << "launch: needs to dynamically allocate " << num_dynamic_smem_bytes << " bytes" << std::endl;
}

