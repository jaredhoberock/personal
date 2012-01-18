#pragma once

#include <thrust/detail/type_traits.h>

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 1
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 2
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 3
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 4
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 5
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 6
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args),
               thrust::get<5>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 7
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args),
               thrust::get<5>(args),
               thrust::get<6>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 8
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args),
               thrust::get<5>(args),
               thrust::get<6>(args),
               thrust::get<7>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 9
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args),
               thrust::get<5>(args),
               thrust::get<6>(args),
               thrust::get<7>(args),
               thrust::get<8>(args));
}

template<typename T, typename Tuple>
__host__ __device__
  typename thrust::detail::enable_if<
    thrust::tuple_size<Tuple>::value == 10
  >::type
  tuple_placement_new(void *ptr, const Tuple &args)
{
  ::new(ptr) T(thrust::get<0>(args),
               thrust::get<1>(args),
               thrust::get<2>(args),
               thrust::get<3>(args),
               thrust::get<4>(args),
               thrust::get<5>(args),
               thrust::get<6>(args),
               thrust::get<7>(args),
               thrust::get<8>(args),
               thrust::get<9>(args));
}


