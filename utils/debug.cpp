/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#define BOOST_STACKTRACE_USE_ADDR2LINE

#include <cmath>
#include "awnn/logging.h"
#include "utils/debug.h"
#ifdef USE_BOOST_STACKTRACE
#include <boost/stacktrace.hpp>
#endif
#include <iostream>
#include <csignal>     // ::signal, ::raise

void print_trace(void) {
#ifdef USE_BOOST_STACKTRACE
  std::cout << boost::stacktrace::stacktrace();
#else
  PINF("Boost stack trace not available");
  return;
#endif
}

int list_get_count(struct list_head *head) {
  struct list_head *pos;
  int count = 0;
  list_for_each(pos, head) {
    PSTR("[%d]%p\t", count, pos);
    ++count;
  }
  PSTR("\n");
  return count;
}

void my_signal_handler(int signum) {
    ::signal(signum, SIG_DFL);
    PWRN("Segfault:"); 
    print_trace();
    PWRN("now exiting...");
    exit(-1);
}

void init_helper_env(){
	::signal(SIGSEGV, &my_signal_handler);
	::signal(SIGABRT, &my_signal_handler);

  // print stacktrace during faults
  PMAJOR("Sig handler enabled");
}


clocktime_t get_clocktime(){
  clocktime_t t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t;
}
double get_elapsed_ms(clocktime_t start, clocktime_t end){
  double elapsed = (end.tv_sec - start.tv_sec);
  elapsed += (end.tv_nsec - start.tv_nsec)/1000000000.0;
  return elapsed*1000;
}
