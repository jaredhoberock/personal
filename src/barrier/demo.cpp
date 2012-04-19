#include <iostream>
#include <csetjmp>
#include <vector>
#include <cstring>

namespace this_thread_group
{


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

int &current_thread_id()
{
  static int tid;
  return tid;
}

int next_current_thread_id()
{
  return (current_thread_id()+1) % thread_state.size();
}

void set_current_thread_id(int id)
{
  current_thread_id() = id;
}

void set_next_current_thread_id()
{
  current_thread_id() = next_current_thread_id();
}

void barrier()
{
  std::clog << "entering barrier from thread " << current_thread_id() << std::endl;

  // save this thread's state
  int returning_thread = setjmp(thread_state[current_thread_id()]);

  if(!returning_thread)
  {
    // switch to the next ready thread
    std::clog << "jumping from thread " << current_thread_id() << " to thread " << next_current_thread_id() << std::endl;

    set_next_current_thread_id();
    std::longjmp(thread_state[current_thread_id()], current_thread_id() + 1);
  }
  else
  {
    std::clog << "barrier(): result of save is " << returning_thread << std::endl;
    --returning_thread;

    set_current_thread_id(returning_thread);

    std::clog << "jumped into thread " << current_thread_id() << std::endl;
  }

  std::clog << "thread " << current_thread_id() << " exiting barrier()" << std::endl;
} // end barrier


}


template<typename Function>
  void launch(int num_threads, Function f)
{
  // save the start state
  this_thread_group::state start_state;
  if(!setjmp(start_state))
  {
    std::clog << "initializing start state" << std::endl;

    // init each thread's state to the start state
    this_thread_group::thread_state.clear();
    this_thread_group::thread_state.resize(num_threads, start_state);

    this_thread_group::set_current_thread_id(0);
  }
  else
  {
    // new thread
    std::clog << "jumped to thread "<< this_thread_group::current_thread_id() << " start state into thread " << std::endl;
  }

  // execute the thread
  f();

  std::clog << "done with thread " << this_thread_group::current_thread_id() << std::endl;

  this_thread_group::barrier();
}


namespace this_thread
{

int id()
{
  return this_thread_group::current_thread_id();
}

}


void foo()
{
  std::cout << "hello, world from thread " << this_thread::id() << std::endl;

  this_thread_group::barrier();

  std::cout << "after barrier in thread " << this_thread::id() << std::endl;
}

int main()
{
  launch(10, foo);

  std::cout << "main(): back from launch" << std::endl;

  return 0;
}

