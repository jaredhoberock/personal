#pragma once

template<typename T>
  class shared
{
  public:
    shared() = delete;
    shared(const shared &) = delete;

  private:
    friend class launch_core_access;
};

