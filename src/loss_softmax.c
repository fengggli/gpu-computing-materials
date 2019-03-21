#include "awnn/loss_softmax.h"
#include "awnn/logging.h"
#include <math.h>

status_t loss_softmax(tensor_t const x, label_t const real_labels[], T *ptr_loss, tensor_t dx){
  status_t ret = S_ERR;

  tensor_t scores = tensor_make_copy(x);
  tensor_t exps = tensor_make_copy(x); // save the exponentials
  const uint cnt_imgs = scores.dim.dims[0];
  const uint cnt_features = scores.dim.dims[1];

  T loss = 0;

  for(uint i_img = 0 ; i_img < cnt_imgs; ++i_img){
    label_t this_label = real_labels[i_img];
    T max_score = T_MIN; // initialize to a small number
    T sum_of_exp = 0;

    // get the maximun score
    for(uint i_feature = 0; i_feature < cnt_features; ++i_feature){
      if(scores.data[i_img * cnt_features + i_feature] > max_score) max_score = scores.data[i_img * cnt_features + i_feature];
    }
    for(uint i_feature = 0; i_feature < cnt_features; ++i_feature){
      scores.data[i_img * cnt_features + i_feature] -= max_score; // substract maximum for numerical stability
      sum_of_exp += exp(scores.data[i_img * cnt_features + i_feature]);
      PINF("sum_exp += %lf", exp(scores.data[i_img * cnt_features + i_feature]));
    }
    loss += log(sum_of_exp) - scores.data[this_label];
  }

  *ptr_loss = loss/cnt_imgs;
  ret = S_OK;

  tensor_destroy(scores);
  tensor_destroy(exps);

  return ret;
}


