#include "launch.hpp"
#include "shared.hpp"
#include <iostream>

struct bar
{
  float x, y, z;
};

void foo(double x, float y, shared<int> &z, shared<bar> &w)
{
}

int main()
{
  launch(foo, 10., 13.f, 13, bar());

  std::cout << "shared size should be " << sizeof(int) + sizeof(bar) << std::endl;

  return 0;
}

