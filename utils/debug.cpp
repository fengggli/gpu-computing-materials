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
#include <iostream>

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
