#include "awnn/solver.h"
#include "utils/data_cifar.h"

static uint _get_correct_count(tensor_t const x, label_t const *labels,
                               uint nr_record, model_t const *model) {
  AWNN_CHECK_EQ(x.dim.dims[0], nr_record);
  tensor_t scores = mlp_forward_infer(model, x);
  label_t label_predicted[nr_record];

  uint nr_classes = scores.dim.dims[1];
  for (uint i = 0; i < nr_record; i++) {
    uint predicted_cls_id = -1;
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
 * get accuracy of a batch of data
 */
double check_accuracy(data_loader_t *loader, uint val_sz, uint batch_sz,
                      model_t const *model) {
  uint nr_correct = 0;
  uint nr_total = 0;

  tensor_t x_val;
  label_t *labels_val;

  uint nr_iterations = val_sz / batch_sz;

  for (uint iteration = 0; iteration < nr_iterations; iteration++) {
    uint nr_record =
        get_validation_batch(loader, &x_val, &labels_val, iteration, batch_sz);
    nr_correct += _get_correct_count(x_val, labels_val, nr_record, model);
    nr_total += nr_record;
  }

  double accuracy = (nr_correct + 0.0) / nr_total;
  PINF("[Accuracy]: %.3f, [%u/%u]", accuracy, nr_correct, nr_total);
  return accuracy;
}
