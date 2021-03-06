//
// Created by lifen on 4/5/19.
//

#include <stdio.h>
#include <string>
#include "awnn/memory.h"
#include "awnn/tensor.h"
#include "utils/data_cifar.h"

// This read all data into memory and calculate offset of each batch

const uint C = 3;
const uint H = 32;
const uint W = 32;

// cifar 10 training set will be split into train/val
static const uint nr_train_img = 50000;
static const uint nr_test_img = 10000;

// default train/val split, can be overwritten with cifar_split_train
static const uint nr_default_train_sz = 49000;
static const uint nr_default_val_sz = 1000;

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
  for (uint i = 0; i < nr_elem; i++) {
    uint channel_id = i / (H * W);
    // buffer_float[i] = T(buffer_str[i]);
    // normalize to (0, 255) -> (0, 1), then rescale to N(0,1)
    double byte_value = ((unsigned char)(buffer_str[i]) / 255.0);
    double this_value =
        (byte_value - channel_mean[channel_id]) / channel_std[channel_id];
    // PINF("value %.3f normalized to %.3f", byte_value, this_value);
    buffer_float[i] = this_value;
  }
}

status_t cifar_open_batched(data_loader_t *loader, const char *input_folder,
                            int batch_sz, int nr_readers) {
  label_t label;
  uint bytes_per_img = C * H * W;
  char *buffer_str = (char *)mem_alloc(bytes_per_img);
  char inFileName[MAX_STR_LENGTH];

  AWNN_CHECK_GT(nr_readers, 0);
  loader->nr_readers = nr_readers;
  loader->readers_info = (struct reader_local_info *)mem_zalloc(
      sizeof(reader_local_info) * nr_readers);

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

  for (uint itemid = 0; itemid < nr_train_img; ++itemid) {
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
    PERR("file %s not found \n check data/cifar10/get_cifar10.sh", inFileName);
    exit(-1);
  }

  for (uint itemid = 0; itemid < nr_test_img; ++itemid) {
    read_image(data_file, &label, buffer_str);
    loader->label_test[itemid] = label;
    preprocess_data(buffer_str, &loader->data_test[itemid * bytes_per_img],
                    bytes_per_img);
  }
  fclose(data_file);

  mem_free(buffer_str);

  // by default set train/val split
  cifar_split_train(loader, nr_default_train_sz, nr_default_val_sz);

  if (batch_sz > 0) {
    loader->batch_sz = uint(batch_sz);
    PINF("Cifar data loader batch size %d", batch_sz);
  } else {
    PWRN("Cifar data loader without batch size is deprecated[19-11-28]");
  }

  return S_OK;
}

status_t cifar_open(data_loader_t *loader, const char *input_folder) {
  return cifar_open_batched(loader, input_folder, 0, 1);
}

status_t cifar_split_train(data_loader_t *loader, uint train_sz,
                           uint validation_sz) {
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
  mem_free(loader->readers_info);

  return S_OK;
}

// Finer get
uint get_train_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                     uint batch_id, uint batch_sz) {
  uint i_start = batch_id * batch_sz;
  uint i_end = i_start + batch_sz;

  if (i_end > loader->train_split) i_end = loader->train_split;
  uint nr_imgs = i_end - i_start;

  uint shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_train + i_start * (C * H * W);
  *label = loader->label_train + i_start;
  return nr_imgs;
}

uint get_train_batch_mt(data_loader_t *loader, uint thread_id) {
  int nr_threads = loader->nr_readers;

  AWNN_CHECK_GT(loader->batch_sz, 0);

  // thread local info
  struct reader_local_info *reader_info = loader->readers_info + thread_id;
  tensor_t *x = &(reader_info->cur_x);
  label_t **label = &(reader_info->cur_label);

  uint batch_id = reader_info->cur_train_batch;
  uint batch_sz = loader->batch_sz;

  uint i_start = batch_id * batch_sz;
  uint i_end = i_start + batch_sz;

  if (i_end > loader->train_split) i_end = loader->train_split;
  uint max_imgs_per_thread = (i_end - i_start + nr_threads - 1) / nr_threads;
  uint nr_imgs;
  if (thread_id == nr_threads - 1 && (i_end - i_start) % nr_threads) {
    nr_imgs = (i_end - i_start) % max_imgs_per_thread;
  } else {
    nr_imgs = max_imgs_per_thread;
  }

  uint shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_train + (i_start + thread_id*nr_imgs) * (C * H * W);
  *label = loader->label_train + (i_start + thread_id*nr_imgs);
  return nr_imgs;
}

// Finer get
uint get_validation_batch(data_loader_t const *loader, tensor_t *x,
                          label_t **label, uint batch_id, uint batch_sz) {
  uint i_start = batch_id * batch_sz + loader->val_split;
  uint i_end = i_start + batch_sz;

  if (i_end > nr_train_img) i_end = nr_train_img;
  uint nr_imgs = i_end - i_start;

  uint shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_train + i_start * (C * H * W);
  *label = loader->label_train + i_start;
  return nr_imgs;
}

uint get_test_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                    uint batch_id, uint batch_sz) {
  uint i_start = batch_id * batch_sz;
  uint i_end = i_start + batch_sz;

  if (i_end > nr_test_img) i_end = nr_test_img;
  uint nr_imgs = i_end - i_start;

  uint shape_batch[] = {nr_imgs, C, H, W};
  x->dim = make_dim_from_arr(4, shape_batch);
  x->data = loader->data_test + i_start * (C * H * W);
  *label = loader->label_test + i_start;
  return nr_imgs;
}
