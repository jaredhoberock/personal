#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/tbb/vector.h>
#include <thrust/logical.h>
#include <algorithm>
#include <iostream>

// see http://en.wikipedia.org/wiki/Associativity#Definition

template<typename ForwardIterator, typename BinaryFunction>
  bool is_associative(ForwardIterator first, ForwardIterator last, BinaryFunction op)
{
  using namespace thrust;

  typedef typename iterator_value<ForwardIterator>::type value_type;
  
  return all_of(first, last, [op](value_type x){
    all_of(first, last, [x,op](value_type y){
      all_of(first, last, [x,y,op](value_type z){
        return op(op(x,y), z) == op(x, op(y,z));
      });
    });
  });
}

int main()
{
  const size_t n = 1000;

  thrust::tbb::vector<int> int_vec(n);
  thrust::tbb::vector<float> float_vec(n);

  // use serial std::generate with rand
  std::generate(int_vec.begin(), int_vec.end(), std::rand);
  std::generate(float_vec.begin(), float_vec.end(), rand01);

  // integer addition is associative
  std::cout << "is_associative(int,plus): " << is_associative(int_vec.begin(), int_vec.end(), std::plus<int>()) << std::endl;

  // integer subtraction is not associative
  std::cout << "is_associative(int,minus): " << is_associative(int_vec.begin(), int_vec.end(), std::minus<int>()) << std::endl;

  // floating point addition is not necessarily associative
  std::cout << "is_associative(float,plus): " << is_associative(float_vec.begin(), float_vec.end(), std::plus<float>()) << std::endl;

  return 0;
}

