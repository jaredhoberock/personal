#pragma once

template<typename... T> struct type_list {};

template<typename> struct type_list_head;

template<typename Head, typename... Tail>
  struct type_list_head<type_list<Head,Tail...>>
{
  typedef Head type;
};

template<typename> struct type_list_tail;

template<typename Head, typename... Tail>
  struct type_list_tail<type_list<Head, Tail...>>
{
  typedef type_list<Tail...> type;
};


template<unsigned int i, typename> struct type_list_element;

template<typename Head, typename... Tail>
  struct type_list_element<0, type_list<Head, Tail...>>
{
  typedef Head type;
};

template<unsigned int i, typename Head, typename... Tail>
  struct type_list_element<i, type_list<Head, Tail...>>
    : type_list_element<i-1, type_list<Tail...>>
{};

