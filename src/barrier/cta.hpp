#pragma once

class cta;

namespace this_thread_group
{

// XXX should be thread_local
// XXX should be a stack for nested launch
cta *__singleton;

}

class cta
{
  public:
    cta()
    {
      this_thread_group::__singleton = this;
    }

    virtual ~cta(){}

    int current_thread_id()
    {
      return m_current_thread_id;
    }

    virtual void barrier() = 0;

    virtual int num_threads() = 0;

  protected:
    int next_current_thread_id()
    {
      return (current_thread_id()+1) % num_threads();
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
    int m_current_thread_id;
};

namespace this_thread_group
{

int current_thread_id()
{
  return __singleton->current_thread_id();
}

void barrier()
{
  __singleton->barrier();
}

} // end this_thread_group

