#include <vector>
#include <iostream>
#include <functional>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include "is_strict_weak_ordering.hpp"

int main()
{
  std::vector<int> int_vec(1000);

  std::generate(int_vec.begin(), int_vec.end(), std::rand);

  // this should pass
  std::cout << "is_strict_weak_ordering(int_vec): " << is_strict_weak_ordering(int_vec.begin(), int_vec.end(), std::less<int>()) << std::endl;

  
  std::vector<float> float_vec(1000);

  std::generate(float_vec.begin(), float_vec.end(), std::rand);

  // add a NaN to make the test fail
  float_vec[5] = std::numeric_limits<float>::quiet_NaN();

  // this should fail
  std::cout << "is_strict_weak_ordering(float_vec): " << is_strict_weak_ordering(float_vec.begin(), float_vec.end(), std::less<float>()) << std::endl;

  return 0;
}

