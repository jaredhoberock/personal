#pragma once

struct null_type {};

template<
  typename T0 = null_type,
  typename T1 = null_type, typename T2 = null_type, typename T3 = null_type,
  typename T4 = null_type, typename T5 = null_type, typename T6 = null_type,
  typename T7 = null_type, typename T8 = null_type, typename T9 = null_type
>
  struct type_list {};

template<typename> struct type_list_head;

template<
  typename Head,
  typename T1 = null_type, typename T2 = null_type, typename T3 = null_type,
  typename T4 = null_type, typename T5 = null_type, typename T6 = null_type,
  typename T7 = null_type, typename T8 = null_type, typename T9 = null_type
>
  struct type_list_head<
    type_list<Head,T1,T2,T3,T4,T5,T6,T7,T8,T9>
  >
{
  typedef Head type;
};

template<
  typename Head,
  typename T1 = null_type, typename T2 = null_type, typename T3 = null_type,
  typename T4 = null_type, typename T5 = null_type, typename T6 = null_type,
  typename T7 = null_type, typename T8 = null_type, typename T9 = null_type
>
  struct type_list_tail<
    type_list<Head,T1,T2,T3,T4,T5,T6,T7,T8,T9>
  >
{
  typedef type_list<T1,T2,T3,T4,T5,T6,T7,T8,T9> type;
};

template<unsigned int i, typename> struct type_list_element;

template<
  typename Head,
  typename T1 = null_type, typename T2 = null_type, typename T3 = null_type,
  typename T4 = null_type, typename T5 = null_type, typename T6 = null_type,
  typename T7 = null_type, typename T8 = null_type, typename T9 = null_type
>
  struct type_list_element<0, type_list<Head,T1,T2,T3,T4,T5,T6,T7,T8,T9> >
{
  typedef Head type;
};

template<
  unsigned int i,
  typename Head,
  typename T1 = null_type, typename T2 = null_type, typename T3 = null_type,
  typename T4 = null_type, typename T5 = null_type, typename T6 = null_type,
  typename T7 = null_type, typename T8 = null_type, typename T9 = null_type
>
  struct type_list_element<i, type_list<Head,T1,T2,T3,T4,T5,T6,T7,T8,T9> >
    : type_list_element<i-1, type_list<T1,T2,T3,T4,T5,T6,T7,T8,T9> >
{};

