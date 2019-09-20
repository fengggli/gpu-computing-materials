#include "awnn/net.h"
#include "awnn/solver.h"
#include "utils/data_cifar.h"

enum{
  CLS_ID_NULL = 10000
};

void sgd_update(param_t *p_param, T learning_rate) {
  tensor_t param = p_param->data;
  tensor_t dparam = p_param->diff;
  T *pelem;
  uint ii;
  AWNN_CHECK_GT(learning_rate, 0);

  tensor_for_each_entry(pelem, ii, dparam) { (*pelem) *= learning_rate; }
  tensor_elemwise_op_inplace(param, dparam, TENSOR_OP_SUB);
  PDBG("updating %s complete.", p_param->name);
}

void do_sgd_update_momentum(tensor_t param, tensor_t dparam, tensor_t velocity, T learning_rate, T momentum) {
  T *pelem;
  uint ii;
  AWNN_CHECK_GT(learning_rate, 0);

  tensor_for_each_entry(pelem, ii, dparam) { (*pelem) *= learning_rate; }
  tensor_for_each_entry(pelem, ii, velocity) { (*pelem) *= momentum; }
  tensor_elemwise_op_inplace(velocity, dparam, TENSOR_OP_SUB);

  tensor_elemwise_op_inplace(param, velocity, TENSOR_OP_ADD);
  PDBG("updating %s complete.", p_param->name);
}

/* support old version*/
void sgd_update_momentum(param_t *p_param, T learning_rate, T momentum) {
  tensor_t param = p_param->data;
  tensor_t dparam = p_param->diff;
  tensor_t *ptr_velocity = &(p_param->velocity);
  if (p_param->velocity.mem_type == EMPTY_MEM) {
    *ptr_velocity = tensor_make_zeros_alike(param);
  }
  tensor_t velocity = *ptr_velocity;
  do_sgd_update_momentum(param, dparam, velocity, learning_rate, momentum);
}

static uint _get_correct_count(tensor_t const x, label_t const *labels,
                               uint nr_record, model_t const *model,
                               tensor_t (*func_forward_infer)(model_t const *,
                                                              tensor_t)) {
  AWNN_CHECK_EQ(x.dim.dims[0], nr_record);
  tensor_t scores = func_forward_infer(model, x);
  label_t label_predicted[nr_record];

  uint nr_classes = scores.dim.dims[1];
  for (uint i = 0; i < nr_record; i++) {
    uint predicted_cls_id = CLS_ID_NULL;
    T max_score = -1000;

    for (uint j = 0; j < nr_classes; j++) {
      T this_score = scores.data[i * nr_classes + j];
      if (this_score > max_score) {
        max_score = this_score;
        predicted_cls_id = j;
      }
    }
    label_predicted[i] = predicted_cls_id;
  }
  // get the accuracy
  uint nr_correct = 0;
  for (uint i = 0; i < nr_record; i++) {
    if (labels[i] == label_predicted[i]) nr_correct++;
  }
  return nr_correct;
}
/*
 * get accuracy of validation data
 */
double check_val_accuracy(data_loader_t *loader, uint val_sz, uint batch_sz,
                          model_t const *model,
                          tensor_t (*func_forward_infer)(model_t const *,
                                                         tensor_t)) {
  uint nr_correct = 0;
  uint nr_total = 0;

  tensor_t x_val;
  label_t *labels_val;

  uint nr_iterations = val_sz / batch_sz;

  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    uint nr_record =
        get_validation_batch(loader, &x_val, &labels_val, iteration, batch_sz);
    nr_correct += _get_correct_count(x_val, labels_val, nr_record, model,
                                     func_forward_infer);
    nr_total += nr_record;
  }

  double accuracy = (nr_correct + 0.0) / nr_total;
  PINF("[Val Accuracy]: %.3f, [%u/%u]", accuracy, nr_correct, nr_total);
  return accuracy;
}

/*
 * Get accuracy of a sample of train data
 */
double check_train_accuracy(data_loader_t *loader, uint sample_sz,
                            uint batch_sz, model_t const *model,
                            tensor_t (*func_forward_infer)(model_t const *,
                                                           tensor_t)) {
  uint nr_correct = 0;
  uint nr_total = 0;

  tensor_t x_train_sampled;
  label_t *labels_train_sampled;

  uint train_sz = loader->train_split;
  uint iterations_per_epoch =
      train_sz / batch_sz;  // how many batches in each epoch
  uint nr_iterations = sample_sz / batch_sz;

  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    // each time choose a random batch
    uint batch_id = (uint)rand() % iterations_per_epoch;
    // PINF("[---traning accuracy] [%u, %u)", batch_id, batch_id + 1);
    uint nr_record = get_train_batch(loader, &x_train_sampled,
                                     &labels_train_sampled, batch_id, batch_sz);
    nr_correct += _get_correct_count(x_train_sampled, labels_train_sampled,
                                     nr_record, model, func_forward_infer);
    nr_total += nr_record;
    if (nr_record < batch_sz) break;
  }

  double accuracy = (nr_correct + 0.0) / nr_total;
  PINF("[train Accuracy]: %.3f, [%u/%u]", accuracy, nr_correct, nr_total);
  return accuracy;
}
