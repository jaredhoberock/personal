#pragma once

template<typename T>
  class shared
{
  public:
    shared(const shared &) = delete;

  private:
    template<typename> friend class launch_core_access;

    shared() {}
};

template<typename T>
  struct launch_core_access
{
  static shared<T> &shared_dummy()
  {
    static shared<T> dummy;
    return dummy;
  }
};

