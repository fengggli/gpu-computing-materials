/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include <cstdio>

template<size_t SIZE, class T> inline size_t array_size(T (&arr)[SIZE]) {
  return SIZE;
}

template <size_t SIZE, class T> inline size_t dim_of_shape(T const (&shape)[SIZE]) {
  return SIZE;
}
