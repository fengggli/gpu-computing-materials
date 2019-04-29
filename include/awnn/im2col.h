/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef IM2COL_H_
#define IM2COL_H_

#include "awnn/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif
T* alloc_col_buffer(uint C, uint HH, uint WW, uint Hout, uint Wout);
void free_col_buffer(T* col_buffer);

void im2col_cpu(T* data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad, const int stride, T* data_col);
void col2im_cpu(T* data_col, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad, const int stride, T* data_im);

#ifdef __cplusplus
}
#endif
#endif
