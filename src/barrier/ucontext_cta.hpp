#include "cta.hpp"
#include <ucontext.h>

class ucontext_cta
  : public cta
{
  public:
    template<typename Function>
      ucontext_cta(int num_threads, Function f)
        : cta()
    {
      thread_state.clear();

      void (*fp)(ucontext_cta *, Function) = execute_thread<Function>;

      // save the return state
      state join_state;
      getcontext(&join_state);
      if(thread_state.empty())
      {
        thread_state.clear();
        thread_state.resize(num_threads);

        for(int i = 0; i < thread_state.size(); ++i)
        {
          getcontext(&thread_state[i]);
          thread_state[i].uc_link = &join_state;
          thread_state[i].uc_stack.ss_sp = thread_state[i].stack;
          thread_state[i].uc_stack.ss_size = sizeof(thread_state[i].stack);
          makecontext(&thread_state[i], (void(*)())fp, 2, this, f);
        }

        // start thread 0
        set_current_thread_id(0);
        setcontext(&thread_state[0]);
      }

      // null the current cta
      this_thread_group::__singleton = 0;
    }

    virtual int num_threads()
    {
      return thread_state.size();
    }

    void barrier()
    {
      // switch to next thread
      int old_thread_id = set_next_current_thread_id();
      swapcontext(&thread_state[old_thread_id], &thread_state[current_thread_id()]);
    }

  private:
    void at_exit()
    {
      barrier();
    }

    template<typename Function>
      static void execute_thread(ucontext_cta *cta, Function f)
    {
      f();

      cta->at_exit();
    }

    struct state
      : ucontext_t
    {
      char stack[1<<16];
    };

    std::vector<state> thread_state;
};

