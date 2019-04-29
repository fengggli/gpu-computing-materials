/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/logging.h"
#include "awnn/tensor.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <chrono>

template<uint SIZE, class T> inline uint array_size(T (&)[SIZE]) {
  return SIZE;
}

template <uint SIZE, class T> inline uint dim_of_shape(T const (&)[SIZE]) {
  return SIZE;
}

/*
 * @breif calculate gradient at x of y=func(x)using numercial approach
 *
 * @param a functor, used as func(input, output);
 * @param x
 * @param dy [input] gradient of y
 * @param dx [output] the gradient of x
 * @param h  [optional input] precision
 *
 * Note: all the tensor needs to be preallocated
 */
status_t eval_numerical_gradient(
    std::function<void(tensor_t const, tensor_t)> const &func, tensor_t const x,
    tensor_t const dy, tensor_t dx, double h = 1e-5) {
  tensor_t y_pos, y_neg;
  y_pos = tensor_make_alike(dy);
  y_neg = tensor_make_alike(dy);

  // func(x, y_pos);
  // return S_OK;
  uint capacity = tensor_get_capacity(x);
  for (uint i = 0; i < capacity; i++) {
    // PINF("[--numerical check]: [%d ] (%u, %u)",i/(capacity/100) , i, capacity);
    PDBG("\n\n===================================================");
    PDBG("calculating dx at flat position [%d]...", i);
    T old_value = x.data[i];

    x.data[i] = old_value + h;
    func(x, y_pos);

    // second iteration the negative takes long
    x.data[i] = old_value - h;
    func(x, y_neg);

    x.data[i] = old_value;

#ifdef CONFIG_DEBUG
    PDBG("###: dumping y_positive:");
    tensor_dump(y_pos);
    PDBG("###: dumping y_negative:");
    tensor_dump(y_neg);
#endif

    tensor_t tmp = y_pos; // shadow copy
    tensor_elemwise_op_inplace(tmp, y_neg, TENSOR_OP_SUB);
    tensor_elemwise_op_inplace(tmp, dy, TENSOR_OP_MUL);

    T partial_deriv = tensor_get_sum(tmp) / (2 * h);
    dx.data[i] = partial_deriv;
    PDBG("Seting dx[%u] = %.7f", i, partial_deriv);
  }
  tensor_destroy(&y_pos);
  tensor_destroy(&y_neg);

  return S_OK;
}

using namespace std::chrono;
using time_point_t = high_resolution_clock::time_point;

void get_cur_time(time_point_t &t){
  t = high_resolution_clock::now();
}

inline time_point_t get_timepoint() {
  return high_resolution_clock::now();
}

void print_time_in_s(time_point_t &begin, time_point_t &end){
  PINF("Time difference = %.3f  ms",((double)duration_cast<microseconds>(end - begin).count())/1000000);
}

void print_time_in_ms(time_point_t &begin, time_point_t &end){
  PINF("Time difference = %.3f  ms",((double)duration_cast<microseconds>(end - begin).count())/1000);
}

void print_time_in_ns(time_point_t &begin, time_point_t &end){
  PINF("Time difference = %lu ns",duration_cast<nanoseconds>(end - begin).count());
}
