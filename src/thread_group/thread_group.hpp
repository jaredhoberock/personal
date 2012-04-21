#pragma once

namespace test
{

class thread_group;

namespace this_thread_group
{

__thread thread_group *__singleton;

} // end this_thread_group

class thread_group
{
  public:
    thread_group(int id)
      : m_id(id),m_current_thread_id(-1)
    {
      this_thread_group::__singleton = this;
    }

    virtual ~thread_group(){}

    int current_thread_id()
    {
      return m_current_thread_id;
    }

    virtual void barrier() = 0;

    virtual int size() = 0;

    int get_id() const
    {
      return m_id;
    }

  protected:
    int next_current_thread_id()
    {
      return (current_thread_id()+1) % size();
    }

    void set_current_thread_id(int id)
    {
      m_current_thread_id = id;
    }

    int set_next_current_thread_id()
    {
      int old_id = current_thread_id();
      m_current_thread_id = next_current_thread_id();
      return old_id;
    }

  private:
    int m_id;
    int m_current_thread_id;
}; // end thread_group

namespace this_thread_group
{

int current_thread_id()
{
  return __singleton->current_thread_id();
}

int get_id()
{
  return __singleton->get_id();
}

void barrier()
{
  __singleton->barrier();
}

} // end this_thread_group

namespace this_thread
{

int get_id()
{
  return this_thread_group::current_thread_id();
}

} // end namespace this_thread

} // end namespace test

#include "detail/ucontext_thread_group.hpp"

