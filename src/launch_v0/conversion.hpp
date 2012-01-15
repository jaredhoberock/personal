#pragma once

struct converts_to_anything
{
  template<typename T> operator T& () const;
  template<typename T> operator const T& () const;
};

template<typename T>
  struct only_converts_to_type
{
  operator T& () const;
  operator const T& () const;
};

template<template<typename> class Template>
  struct only_converts_to_template1
{
  template<typename T> operator Template<T> & () const;
  template<typename T> operator const Template<T> & () const;

  only_converts_to_template1() = delete;
  only_converts_to_template1(const only_converts_to_template1 &) = delete;
};

template<template<typename,typename> class Template>
  struct only_converts_to_template2
{
  template<typename T1, typename T2> operator Template<T1,T2> & () const;
  template<typename T1, typename T2> operator const Template<T1,T2> & () const;

  only_converts_to_template2() = delete;
  only_converts_to_template2(const only_converts_to_template2 &) = delete;
};

