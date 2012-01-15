#pragma once

#include <utility>

template<typename T>
  class shared
{
  public:
    shared(const shared &) = delete;
    shared() = delete;

  private:
    template<typename> friend class launch_core_access;

    // allows friends to construct dummy shared<T> objects
    struct construct_dummy_tag {};
    shared(construct_dummy_tag) {}

    template<typename... Args>
      shared(void *p, Args&&... args)
        : ptr(reinterpret_cast<T*>(p))
    {
      // construct the object given the args
      ::new(static_cast<void*>(ptr)) T(std::forward<Args>(args)...);
    }

    ~shared()
    {
      // destroy the object
      ptr->~T();
    }

    T *ptr;
};

template<typename T>
  struct launch_core_access
{
  static shared<T> &shared_dummy()
  {
    static shared<T> dummy((typename shared<T>::construct_dummy_tag()));
    return dummy;
  }
};

