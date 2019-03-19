/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/tensor.h"
#include <algorithm>
#include <cstdio>
#include <functional>

template<size_t SIZE, class T> inline size_t array_size(T (&arr)[SIZE]) {
  return SIZE;
}

template <size_t SIZE, class T> inline size_t dim_of_shape(T const (&shape)[SIZE]) {
  return SIZE;
}

/*
 * @breif calculate gradient at x of y=func(x)using numercial approach
 *
 * @param a functor, used as func(input, output);
 * @param x
 * @param dy [input] gradient of y
 * @param dx [output] the gradient of x
 *
 * Note: all the tensor needs to be preallocated
 */
status_t eval_numerical_gradient(
    std::function<void(tensor_t const, tensor_t)> const func, tensor_t const x,
    tensor_t const dy, tensor_t dx) {
  uint i = 0;
  uint capacity = tensor_get_capacity(x);

  return S_OK;
}
