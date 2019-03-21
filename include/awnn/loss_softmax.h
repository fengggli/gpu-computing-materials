/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#pragma once

#include "awnn/tensor.h"

typedef uint label_t;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief comput the loss and gradient for softmax classification
 *
 * @param x input data, shape (N, M), where x[i,j] is the score for the jth
 * class in the ith input
 * @param real_labels shape (N,), where y[i] is the label for x[i] and 0< y[i]
 * <M
 * @param ptr_loss [output] the scalar of loss
 * @param dx [output], gradient of the loss with respect to x.
 *
 * TODO: label should be some other types than T
 */
status_t loss_softmax(tensor_t const x, label_t const real_labels[],
                      T *ptr_loss, tensor_t dx);

#ifdef __cplusplus
}
#endif
