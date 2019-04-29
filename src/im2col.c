#include "awnn/im2col.h"
#include "awnn/memory.h"
T* alloc_col_buffer(uint C, uint HH, uint WW, uint Hout, uint Wout){
  uint nr_elem = (C*HH*WW)*(Hout*Wout);
  return mem_alloc(nr_elem*sizeof(T));
}

void free_col_buffer(T* col_buffer){
  mem_free(col_buffer);
}

/** Im2col for single image
 * Originally from caffe */
void im2col_cpu(T* data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad, const int stride, T* data_col){
int pad_h = pad;
int pad_w = pad;
int stride_h = stride;
int stride_w = stride;
int dilation_h = 1;
int dilation_w = 1;
 const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/** Col2img for single image*/
void col2im_cpu(T* data_col, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad, const int stride, T* data_im){

int pad_h = pad;
int pad_w = pad;
int stride_h = stride;
int stride_w = stride;
int dilation_h = 1;
int dilation_w = 1;
  // caffe_set(height * width * channels, Dtype(0), data_im);
  memset(data_im, 0, sizeof(T)*height * width * channels);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

