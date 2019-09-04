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
#include <boost/stacktrace.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <csignal>     // ::signal, ::raise

void print_trace(void) {
  std::cout << boost::stacktrace::stacktrace();
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
	if (boost::filesystem::exists("./backtrace.dump")) {
    // there is a backtrace
    std::ifstream ifs("./backtrace.dump");

    boost::stacktrace::stacktrace st = boost::stacktrace::stacktrace::from_dump(ifs);
    std::cout << "Previous run crashed:\n" << st << std::endl;

    // cleaning up
    ifs.close();
    boost::filesystem::remove("./backtrace.dump");
	}

	::signal(SIGSEGV, &my_signal_handler);
	::signal(SIGABRT, &my_signal_handler);

  // print stacktrace during faults
  PMAJOR("Sig handler enabled");
}
