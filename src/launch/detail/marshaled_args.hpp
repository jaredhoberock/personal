#pragma once

#include <thrust/tuple.h>

namespace detail
{


template<typename ArgTuple>
  class marshaled_args
{
  public:
    __device__
    marshaled_args(void *ptr_to_smem, const ArgTuple &a)
      : m_ptr_to_smem(ptr_to_smem),
        m_args(a)
    {}

    __device__ operator typename thrust::tuple_element<0,ArgTuple>::type () const
    {
      return thrust::get<0>(m_args);
    }

    __device__ void *ptr() const
    {
      return m_ptr_to_smem;
    }

    __device__ const ArgTuple &args() const
    {
      return m_args;
    }

  private:
    void *m_ptr_to_smem;
    ArgTuple m_args;
};


template<typename Arg>
  __device__ marshaled_args<thrust::tuple<const Arg &> > marshal_arg(void *ptr_to_smem, const Arg &arg)
{
  return marshaled_args<thrust::tuple<const Arg &> >(ptr_to_smem, thrust::tie(arg));
} // end marshal_arg()


} // end detail

