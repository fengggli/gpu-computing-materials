#include "awnn/loss_softmax.h"
#include "awnn/logging.h"
#include "awnn/memory.h"
#include <math.h>

label_t *label_make_random(uint nr_elem, uint range){
  label_t *labels = (label_t*)mem_alloc(sizeof(label_t)*nr_elem);
  for(uint i =0; i< nr_elem; i++){
    labels[i] = ((uint)rand())%range;
  }
  return labels;
}

void label_destroy(label_t *labels) {
  mem_free(labels);
}


status_t loss_softmax(tensor_t const x, label_t const * real_labels, T *ptr_loss, awnn_mode_t mode,  tensor_t dx){
  status_t ret = S_ERR;

  tensor_t scores = tensor_make_copy(x);
  const uint cnt_imgs = scores.dim.dims[0];
  const uint cnt_classes = scores.dim.dims[1];

  T loss = 0;

  for(uint i_img = 0 ; i_img < cnt_imgs; ++i_img){
    label_t this_label = real_labels[i_img];
    if(this_label >= cnt_classes){
      PERR("label id should be between [0, %u]", cnt_classes);
      goto end;
    }
    T max_score = T_MIN; // initialize to a small number
    T sum_exp_this_img = 0;

    // get the maximum score
    for(uint i_class = 0; i_class < cnt_classes; ++i_class){
      if(scores.data[i_img * cnt_classes + i_class] > max_score) max_score = scores.data[i_img * cnt_classes + i_class];
    }
    for(uint i_class = 0; i_class < cnt_classes; ++i_class){
      scores.data[i_img * cnt_classes + i_class] -= max_score; // subtract maximum for numerical stability
      T tmp_exp = exp(scores.data[i_img * cnt_classes + i_class]);
      sum_exp_this_img += tmp_exp;

      // fill the gradient
      if(mode != MODE_INFER){
        dx.data[i_img * cnt_classes + i_class] = tmp_exp;
        PDBG("[%u, %u] exp %.3f", i_img, i_class, tmp_exp);
      }
    }

    if(mode != MODE_INFER){
      for(uint i_class = 0; i_class < cnt_classes; ++i_class){
        T point_gradient = dx.data[i_img * cnt_classes + i_class];
        point_gradient/=(sum_exp_this_img);
        if(i_class == this_label){
          point_gradient -= 1.0;
        }
        // PINF("[%u, %u] gradient %.3f", i_img, i_class, point_gradient);
        dx.data[i_img * cnt_classes + i_class] = point_gradient/ cnt_imgs;
      }
    }
    loss += log(sum_exp_this_img) - scores.data[i_img * cnt_classes+ this_label];// -log(e^{y_i}/(\sum_j(e^{y_j}))
  }

#ifdef CONFIG_DEBUG
  PINF("[softmax-internal]: (modified (+/-h) data):");
  tensor_dump(x);
  PINF("[softmax-internal]: gradient");
  tensor_dump(dx);
  PINF("[softmax-internal]: loss: ");
  PINF("%.7f", loss);
#endif

  *ptr_loss = loss/cnt_imgs;
  ret = S_OK;

end:
  tensor_destroy(&scores);
  return ret;
}


