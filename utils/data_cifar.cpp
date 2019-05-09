//
// Created by lifen on 4/5/19.
//

#include <stdio.h>
#include <string>
#include "awnn/memory.h"
#include "awnn/tensor.h"
#include "utils/data_cifar.h"

// This read all data into memory and calculate offset of each batch

const int C = 3;
const int H = 32;
const int W = 32;

// cifar 10 training set will be split into train/val
static const int nr_train_img = 50000;
static const int nr_test_img = 10000;

// default train/val split, can be overwritten with cifar_split_train
static const int nr_default_train_sz = 49000;
static const int nr_default_val_sz = 1000;

// mean value from deep 500 dataset/cifar.py
static const double channel_mean[] = {0.4914, 0.4822, 0.4465};
static const double channel_std[] = {0.2023, 0.1994, 0.2010};

// Read byte stream
void read_image(FILE *file, label_t *label, char *buffer) {
  char label_char;
  fread(&label_char, 1, 1, file);
  *label = label_char;  // has to convert from char
  fread(buffer, 1, (C * H * W), file);
  return;
}

// Saves a float/double
// TODO: substract mean
inline void preprocess_data(char *buffer_str, T *buffer_float, size_t nr_elem) {
  for (int i = 0; (size_t)i < nr_elem; i++) {
    int channel_id = i / (H * W);
    // buffer_float[i] = T(buffer_str[i]);
    // normalize to (0, 255) -> (0, 1), then rescale to N(0,1)
    double byte_value = ((unsigned char)(buffer_str[i]) / 255.0);
    double this_value =
        (byte_value - channel_mean[channel_id]) / channel_std[channel_id];
    // PINF("value %.3f normalized to %.3f", byte_value, this_value);
    buffer_float[i] = this_value;
  }
}

status_t cifar_open(data_loader_t *loader, const char *input_folder) {
  label_t label;
  int bytes_per_img = C * H * W;
  char *buffer_str = (char *)mem_alloc(bytes_per_img);
  char inFileName[MAX_STR_LENGTH];

  /*
   * Training set
   */
  loader->data_train = (T *)mem_alloc(nr_train_img * C * H * W * sizeof(T));
  AWNN_CHECK_NE(NULL, loader->data_train);

  loader->label_train = (label_t *)mem_alloc(nr_train_img * sizeof(label_t));
  AWNN_CHECK_NE(NULL, loader->label_train);

  PINF("Opening Training data");
  // Open files
  snprintf(inFileName, MAX_STR_LENGTH, "%s/data_batch_all.bin", input_folder);

  FILE *data_file = fopen(inFileName, "rb");
  if (data_file == NULL) {
    PERR("file %s not found", inFileName);
    exit(-1);
  }

  for (int itemid = 0; itemid < nr_train_img; ++itemid) {
    read_image(data_file, &label, buffer_str);
    loader->label_train[itemid] = label;
    preprocess_data(buffer_str, &loader->data_train[itemid * bytes_per_img],
                    bytes_per_img);
  }
  fclose(data_file);

  /*
   * Testing set
   */
  loader->data_test = (T *)mem_alloc(nr_test_img * C * H * W * sizeof(T));
  AWNN_CHECK_NE(NULL, loader->data_test);

  loader->label_test = (label_t *)mem_alloc(nr_test_img * sizeof(label_t));
  AWNN_CHECK_NE(NULL, loader->label_test);
  PINF("Opening Testing data");
  // Open files
  snprintf(inFileName, MAX_STR_LENGTH, "%s/test_batch.bin", input_folder);

  data_file = fopen(inFileName, "rb");
  if (data_file == NULL) {
    PERR("file %s not found", inFileName);
    exit(-1);
  }

  for (int itemid = 0; itemid < nr_test_img; ++itemid) {
    read_image(data_file, &label, buffer_str);
    loader->label_test[itemid] = label;
    preprocess_data(buffer_str, &loader->data_test[itemid * bytes_per_img],
                    bytes_per_img);
  }
  fclose(data_file);

  mem_free(buffer_str);

  // by default set train/val split
  cifar_split_train(loader, nr_default_train_sz, nr_default_val_sz);
  return S_OK;
}

status_t cifar_split_train(data_loader_t *loader, int train_sz,
                           int validation_sz) {
  if (train_sz + validation_sz > nr_train_img) {
    PERR("train_sz + val_sz > nr_train_img");
    return S_ERR;
  }
  loader->train_split = train_sz;
  loader->val_split = nr_train_img - validation_sz;
  return S_OK;
}

status_t cifar_close(data_loader_t *loader) {
  mem_free(loader->data_train);
  mem_free(loader->label_train);

  mem_free(loader->data_test);
  mem_free(loader->label_test);
  return S_OK;
}

// Finer get
int get_train_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                     int batch_id, int batch_sz) {
  int i_start = batch_id * batch_sz;
  int i_end = i_start + batch_sz;

  if (i_end > loader->train_split) i_end = loader->train_split;
  int nr_imgs = i_end - i_start;

  int shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_train + i_start * (C * H * W);
  *label = loader->label_train + i_start;
  return nr_imgs;
}

// Finer get
int get_validation_batch(data_loader_t const *loader, tensor_t *x,
                          label_t **label, int batch_id, int batch_sz) {
  int i_start = batch_id * batch_sz + loader->val_split;
  int i_end = i_start + batch_sz;

  if (i_end > nr_train_img) i_end = nr_train_img;
  int nr_imgs = i_end - i_start;

  int shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_train + i_start * (C * H * W);
  *label = loader->label_train + i_start;
  return nr_imgs;
}

int get_test_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                    int batch_id, int batch_sz) {
  int i_start = batch_id * batch_sz;
  int i_end = i_start + batch_sz;

  if (i_end > nr_test_img) i_end = nr_test_img;
  int nr_imgs = i_end - i_start;

  int shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_test + i_start * (C * H * W);
  *label = loader->label_test + i_start;
  return nr_imgs;
}
