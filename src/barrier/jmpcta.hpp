#include "cta.hpp"
#include <csetjmp>
#include <vector>

class jmpcta
  : public cta
{
  public:
    template<typename Function>
      void launch(int num_threads, Function f)
    {
      // save the start state
      state start_state;
      if(!setjmp(start_state))
      {
        std::clog << "launch(): initializing start state" << std::endl;

        // init each thread's state to the start state
        thread_state.clear();
        thread_state.resize(num_threads, start_state);

        set_current_thread_id(0);
      }
      else
      {
        // new thread
        std::clog << "launch(): jumped to thread " << current_thread_id() << " start state into thread " << std::endl;
      }

      // execute the thread
      f();

      std::clog << "launch(): done with thread " << current_thread_id() << std::endl;

      barrier();
    }

    void barrier()
    {
      std::clog << "barrier(): entering barrier from thread " << current_thread_id() << std::endl;

      // save this thread's state
      if(!setjmp(thread_state[current_thread_id()]))
      {
        // switch to the next ready thread
        std::clog << "barrier(): jumping from thread " << current_thread_id() << " to thread " << next_current_thread_id() << std::endl;

        set_next_current_thread_id();
        std::longjmp(thread_state[current_thread_id()], 1);
      }
      else
      {
        std::clog << "barrier(): jumped into thread " << current_thread_id() << std::endl;
      }

      std::clog << "barrier(): thread " << current_thread_id() << " exiting barrier()" << std::endl;
    }

    virtual int num_threads()
    {
      return thread_state.size();
    }

  private:
    struct state
    {
      std::jmp_buf impl;

      operator std::jmp_buf &()
      {
        return impl;
      }

      operator const std::jmp_buf &() const
      {
        return impl;
      }
    };

    std::vector<state> thread_state;
};

