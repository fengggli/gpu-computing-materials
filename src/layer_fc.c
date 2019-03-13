#include "awnn/layer_fc.h"

/* y = x*W+b */
void layer_fc_forward(tensor_t x, tensor_t w, tensor_t b, tensor_t *cache, tensor_t y){
  x_dim = x;

  awnn_mm(x_reshaped, w, y); // y = x*w
  shape_b = tensor_reshape()
  b_extend = tensor_make_scalar()
  awnn_plus(1.0, y, b_extend); // y = y + b

}
void layer_fc_backward(tensor_t dx, tensor_t dw, tensor_t db,tensor_t *cache, tensor_t dy);
