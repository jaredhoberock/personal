#include <cstdio>

#include "uninitialized.hpp"

struct non_trivial_constructor
{
  __device__ non_trivial_constructor()
    : x(13)
  {
    printf("inside non_trivial_constructor with x %d and thread %d\n", x, threadIdx.x);
  }

  int x;
};

struct non_trivial_destructor
{
  __device__ non_trivial_destructor()
  {
    printf("inside non_trivial_destructor with x %d and thread %d\n", x, threadIdx.x);
  }

  int x;
};

struct trivial
{
  int x;
};

__global__ void kernel()
{
  // allow types with trivial constructors & destructors
  __shared__ trivial triv;
  triv.x = 42;
  printf("Trivial types are allowed to be declared shared. triv's value is %d\n", triv.x);

  // disallow types with non-trivial constructor
  // __shared__ non_trivial_constructor y; // Error!

  // disallow types with non-trivial destructor
  // __shared__ non_trivial_destructor z; // Error!

  // wrap __shared__ variables with uninitialized, which has trivial constructor and destructor

  __shared__ uninitialized<non_trivial_constructor> y; // OK!

  // user explicitly constructs the wrapped object
  if(threadIdx.x == 0)
  {
    y.construct();
  }
  __syncthreads();

  __shared__ uninitialized<non_trivial_destructor> z; // OK!

  if(threadIdx.x == 0)
  {
    // access to underlying type
    // the explicit get() is unfortunate
    z.get().x = 13;
  }
  __syncthreads();

  // user explicitly destroys the wrapped object
  if(threadIdx.x == 0)
  {
    z.destroy();
  }
  __syncthreads();
}

int main()
{
  kernel<<<1,1>>>();
  cudaThreadSynchronize();
  return 0;
}

