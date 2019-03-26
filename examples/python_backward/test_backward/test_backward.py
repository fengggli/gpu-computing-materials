from unittest import TestCase
import numpy as np

from python_backward.backward import conv_backward_im2col
from python_backward.test_backward import cache
from python_forward.forward import conv_forward_im2col


class TestConvBackwardIm2col(TestCase):
    def test_conv_backward_im2col(self):

        params = {
            'stride': 2,
            'pad': 1
        }

        nr_img = 2;
        sz_img = 4;
        nr_in_channel = 3;
        sz_filter = 4;
        nr_filter = 3;

        y = [0.02553947, 0.01900658, -0.03984868, -0.09432237,
             0.05964474, 0.09894079, 0.12641447, 0.19823684,
             0.09375, 0.178875, 0.29267763, 0.49079605,
             -0.36098684, -0.57783553, -0.67079605, -1.06632237,
             0.28701316, 0.42294079, 0.41630921, 0.6075,
             0.93501316, 1.42371711, 1.50341447, 2.28132237]

        sz_out = 1 + (sz_img + 2 * params['pad'] - sz_filter) / params['stride']
        shape_y = (nr_img, nr_filter, sz_out, sz_out) # 4x2x5x5

        x_size = nr_img * nr_in_channel * sz_img * sz_img
        w_size = nr_filter * nr_in_channel * sz_filter * sz_filter

        # input for backward
        dy = np.linspace(-0.1, 0.5).reshape(shape_y)
        dx = np.linspace(-.1, .5, x_size).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        dw = np.linspace(-0.2, 0.3, w_size).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

        # dx, dw, db = conv_backward_im2col(dx, dw, cache, dy)  # backward needs to call free_lcache(cache);
        # # EXPECT_EQ(ret, S_OK);
        #
        # # II.Numerical check
        # # I had to make this copy since lambda doesn't allow me to use global variable
        #
        # x_copy = tensor_make_copy(x);
        # w_copy = tensor_make_copy(w);
        #
        # dx_ref = tensor_make_alike(x);
        # dw_ref = tensor_make_alike(w);
        #
        # # evaluate gradient of x
        # eval_numerical_gradient(
        #     [w_copy](tensor_t const in, tensor_t out) {
        #         convolution_forward( in, w_copy, NULL, params, out);
        #     },
        #     x, dy, dx_ref);
        #
        # # EXPECT_LT(tensor_rel_error(dx_ref, dx), 1e-7);
        # # PINF("gradient check of x... is ok");
        #
        # # evaluate gradient of w
        # eval_numerical_gradient(
        #     [x_copy](tensor_t const in, tensor_t out) {
        #         convolution_forward(x_copy, in, NULL, params, out);
        #     },
        #     w, dy, dw_ref);
        # # EXPECT_LT(tensor_rel_error(dw_ref, dw), 1e-7);
        # # PINF("gradient check of w... is ok");
        #
        # # EXPECT_EQ(ret, S_OK)
