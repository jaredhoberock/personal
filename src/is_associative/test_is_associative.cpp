#include <vector>
#include <functional>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include "is_associative.hpp"

float rand01()
{
  return static_cast<float>(std::rand()) / RAND_MAX;
}

bool fuzzy_equals(float x, float y)
{
  return fabs(x - y) <= 1e-5f;
}

int main()
{
  const size_t n = 1000;

  std::vector<int> int_vec(n);
  std::vector<float> float_vec(n);

  std::generate(int_vec.begin(), int_vec.end(), std::rand);
  std::generate(float_vec.begin(), float_vec.end(), rand01);

  // integer addition is associative
  std::cout << "is_associative(int,plus): " << is_associative(int_vec.begin(), int_vec.end(), std::plus<int>()) << std::endl;

  // integer subtraction is not associative
  std::cout << "is_associative(int,minus): " << is_associative(int_vec.begin(), int_vec.end(), std::minus<int>()) << std::endl;

  // floating point addition is not necessarily associative
  std::cout << "is_associative(float,plus): " << is_associative(float_vec.begin(), float_vec.end(), std::plus<float>()) << std::endl;

  // floating point addition may be "fuzzy" associative
  std::cout << "is_associative(float,plus,fuzzy_equals): " << is_associative(float_vec.begin(), float_vec.end(), std::plus<float>(), fuzzy_equals) << std::endl;;

  return 0;
}

