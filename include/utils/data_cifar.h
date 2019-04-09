/*
 * Cifar10 Data loader
 *
 * Original data can be downloaded using data/cifar10/get_cifar10.sh
 * Part of the code is from Caffe.
 */

#pragma once

#include "awnn/common.h"
#include "awnn/data_utils.h"
#include "awnn/tensor.h"

/*#ifdef __cplusplus*/
// extern "C" {
/*#endif*/

// Open data from files
status_t cifar_open(data_loader_t *loader, const char *input_folder);
status_t cifar_close(data_loader_t *loader);


// Split  trainset to train/val
status_t cifar_split_train(data_loader_t *loader, uint train_sz, uint val_sz);

/*
 * Get a train batch
 *
 * @return number of imgs in this this batch
 */
uint get_train_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                     uint batch_id, uint batch_sz);

/*
 * Get a test batch
 *
 * @return number of imgs in this this batch
 */
uint get_test_batch(data_loader_t const *loader, tensor_t *x, label_t **label,
                    uint batch_id, uint batch_sz);

//#ifdef __cplusplus
//}
// extern "C"
/*#endif*/
