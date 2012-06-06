#include <iostream>
#include <algorithm>
#include "../time_invocation/time_invocation.hpp"
#include <cassert>
#include <vector>

int g_i;

inline void fill(int *dst, int val)
{
  dst[g_i] = val;
}

void fill_pass_by_global(int *dst, int val, std::size_t n)
{
  for(int id = 0; id != n; ++id)
  {
    g_i = id;

    fill(dst, val);
  }
}

inline void fill(const int i, int *dst, int val)
{
  dst[i] = val;
}

void fill_pass_by_parameter(int *dst, int val, std::size_t n)
{
  const int num_ids = n;
  for(int id = 0; id != num_ids; ++id)
  {
    fill(id, dst, val);
  }
}

void fill_loop(int *dst, int val, std::size_t n)
{
  int *dst_last = dst + n;
  for(; dst != dst_last; ++dst)
  {
    *dst = val;
  }
}

int main()
{
  std::size_t n = 1 << 23;
  int val = 13;
  std::vector<int> ref(n), dst(n);

  for(std::size_t i = 0; i < n; ++i)
  {
    ref[i] = val;
  }

  std::fill(dst.begin(), dst.end(), 0);
  time_invocation(1, fill_loop, dst.data(), val, n);
  assert(ref == dst);

  double loop_time = time_invocation(1000, fill_loop, dst.data(), val, n);

  std::cout << "loop_fill mean duration: " << loop_time << std::endl;


  std::fill(dst.begin(), dst.end(), 0);
  time_invocation(1, fill_pass_by_global, dst.data(), val, n);
  assert(ref == dst);

  double pass_by_global_time = time_invocation(1000, fill_pass_by_global, dst.data(), val, n);

  std::cout << "pass_by_global mean duration:  " << pass_by_global_time << std::endl;


  std::fill(dst.begin(), dst.end(), 0);
  time_invocation(1, fill_pass_by_parameter, dst.data(), val, n);
  assert(ref == dst);

  double pass_by_parameter_time = time_invocation(1000, fill_pass_by_parameter, dst.data(), val, n);

  std::cout << "pass_by_parameter mean duration:  " << pass_by_parameter_time << std::endl;


  std::cout << "pass by global penalty: " << pass_by_global_time / loop_time << std::endl;
  std::cout << "pass by parameter penalty: " << pass_by_parameter_time / loop_time << std::endl;

  return 0;
}


