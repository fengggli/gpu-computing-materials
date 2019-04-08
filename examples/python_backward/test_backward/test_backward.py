from unittest import TestCase
import numpy as np
import pytest

from python_backward.backward import convolution_backward, tpose1230
from python_backward.test_backward.numerical import eval_numerical_gradient_array
from python_forward.forward import conv_forward


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class TestConvBackward(TestCase):
    """
    https://github.com/fengggli/cs231n-assignments/blob/d4cbe582a794a5b33d81a1ecdb64f1fd3844eaaa/assignment2/ConvolutionalNetworks.ipynb
    """


    def test_bkwrd_from_jupyter_example(self):
        np.random.seed(231)

        x = np.random.randn(4, 3, 5, 5)
        print("x")
        print(x.shape)
        print(list(x.flatten()))

        w = np.random.randn(2, 3, 3, 3)
        print("w")
        print(w.shape)
        print(list(w.flatten()))

        dout = np.random.randn(4, 2, 5, 5) # standin for the derivative from next layer
        print("dout")
        print(dout.shape)
        print(list(dout.flatten()))

        conv_param = {'stride': 1, 'pad': 1}

        dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, conv_param)[0], x, dout)
        print("dx_num")
        print(dx_num.shape)
        print(list(dx_num.flatten()))

        dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, conv_param)[0], w, dout)
        print("dw_num")
        print(dw_num.shape)
        print(list(dw_num.flatten()))

        y, cache = conv_forward(x, w, conv_param)
        print("y")
        print(y.shape)
        print(list(y.flatten()))

        x_cached, w_cached, _, x_cols_cached = cache
        print("x_cached")
        print(x_cached.shape)
        print(list(x_cached.flatten()))

        print("w_cached")
        print(w_cached.shape)
        print(list(w_cached.flatten()))

        print("x_cols_cached")
        print(x_cols_cached.shape)
        print(list(x_cols_cached.flatten()))


        dx, dw = convolution_backward(dout, cache)

        # Your errors should be around e-8 or less.
        print('Testing conv_backward_naive function')
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))


    def test_tpose1230(self):
        """
        this function checks the manual transpose function that reshapes to a 1230
            pattern and compares it to the built in numpy function
        """

        conv_params = {
            'stride': 2,
            'pad': 1
        }

        nr_img = 2;
        sz_img = 4;
        nr_in_channel = 3;
        sz_filter = 4;
        nr_filter = 3;

        a = np.random.randn(2, 1, 3, 2)
        p = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(1, 2, 3, 3)
        x = np.linspace(-.1, .5, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        w = np.linspace(-0.2, 0.3, 3 * 3 * 4 * 6).reshape(3, 3, 4, 6)

        # self.assertEqual(tpose1230(p).all(), p.transpose(1, 2, 3, 0).all())
        # self.assertEqual(tpose1230(w).all(), w.transpose(1, 2, 3, 0).all())
        # self.assertEqual(tpose1230(x).all(), x.transpose(1, 2, 3, 0).all())


        self.assertTrue(np.array_equal(tpose1230(a), a.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(p), p.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(w), w.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(x), x.transpose(1, 2, 3, 0)))

        self.assertEqual(a.shape[0], a.transpose(1, 2, 3, 0).shape[3])
        self.assertEqual(a.shape[1], a.transpose(1, 2, 3, 0).shape[0])
        self.assertEqual(a.shape[2], a.transpose(1, 2, 3, 0).shape[1])
        self.assertEqual(a.shape[3], a.transpose(1, 2, 3, 0).shape[2])

        # print()
        # print(tpose1230(p).flatten())
        # print()
        # print(list(p.transpose(1, 2, 3, 0).flatten()))
        # print()
        # print(list(x.transpose(1, 2, 3, 0).flatten()))
        # print()
        # print(list(w.transpose(1, 2, 3, 0).flatten()))

    @pytest.mark.skip(reason="this test is broken and needs help")
    def test_backward(self):
        conv_params = {
            'stride': 2,
            'pad': 1
        }

        nr_img = 2;
        sz_img = 4;
        nr_in_channel = 3;
        sz_filter = 4;
        nr_filter = 3;

        a = np.random.randn(2, 1, 3, 2)
        p = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(1, 2, 3, 3)
        x = np.linspace(-.1, .5, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        w = np.linspace(-0.2, 0.3, 3 * 3 * 4 * 6).reshape(3, 3, 4, 6)


        x = np.linspace(-.1, .5, x_size).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        w = np.linspace(-0.2, 0.3, w_size).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)
        dout = np.random.randn(nr_img, nr_in_channel - 1, sz_img, 4)  # standin for the derivitive from next layer

        dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, conv_params)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, conv_params)[0], w, dout)

        y, cache = conv_forward(x, w, conv_param=conv_params)
        dx, dw = convolution_backward(dout, cache)

        # self.assertEqual(tpose1230(p).all(), p.transpose(1, 2, 3, 0).all())
        # self.assertEqual(tpose1230(w).all(), w.transpose(1, 2, 3, 0).all())
        # self.assertEqual(tpose1230(x).all(), x.transpose(1, 2, 3, 0).all())


        self.assertTrue(np.array_equal(tpose1230(a), a.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(p), p.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(w), w.transpose(1, 2, 3, 0)))
        self.assertTrue(np.array_equal(tpose1230(x), x.transpose(1, 2, 3, 0)))

        self.assertEqual(a.shape[0], a.transpose(1, 2, 3, 0).shape[3])
        self.assertEqual(a.shape[1], a.transpose(1, 2, 3, 0).shape[0])
        self.assertEqual(a.shape[2], a.transpose(1, 2, 3, 0).shape[1])
        self.assertEqual(a.shape[3], a.transpose(1, 2, 3, 0).shape[2])

        # print()
        # print(tpose1230(p).flatten())
        # print()
        # print(list(p.transpose(1, 2, 3, 0).flatten()))
        # print()
        # print(list(x.transpose(1, 2, 3, 0).flatten()))
        # print()
        print(list(w.transpose(1, 2, 3, 0).flatten()))


    def test_conv_backward_copy_231_assign(self):
        np.random.seed(231)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)

        dout = np.random.randn(4, 2, 5, 5) # standin for the derivitive from next layer
        conv_param = {'stride': 1, 'pad': 1}

        dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, conv_param)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, conv_param)[0], w, dout)

        out, cache = conv_forward(x, w, conv_param)
        dx, dw = convolution_backward(dout, cache)
        # print(dw_num)
        # print(dw)

        # Your errors should be around e-8 or less.
        print('Testing conv_backward_naive function')
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))

    @pytest.mark.skip
    def test_backward_from_picture(self):

        conv_params = {
            'stride': 1,
            'pad': 0
        }

        nr_img = 1
        sz_img = 3
        nr_in_channel = 2  # input channels
        sz_filter = 2
        nr_filter = 2  # num output channels


        conv_params = {
            'stride': 1,
            'pad': 0
        }

        nr_img = 2
        sz_img = 3
        nr_in_channel = 2  # input channels
        sz_filter = 4
        nr_filter = 2  # num output channels

        # x = np.random.randn(nr_img, nr_in_channel, sz_img, 4)
        # w = np.random.randn(nr_filter, nr_in_channel, sz_filter, sz_filter)
        x = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        w = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2]).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

        dout = np.random.randn(nr_img, nr_in_channel - 1, sz_img, 4) # standin for the derivitive from next layer

        dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, conv_params)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, conv_params)[0], w, dout)

        y, cache = conv_forward(x, w, conv_param=conv_params)
        dx, dw = convolution_backward(dout, cache)

        # Your errors should be around e-8 or less.
        print('Testing conv_backward_naive function')
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))

    def test_backward_from_picture(self):
        conv_params = {
            'stride': 1,
            'pad': 0
        }

        nr_img = 2
        sz_img = 3
        nr_in_channel = 2  # input channels
        sz_filter = 4
        nr_filter = 2  # num output channels

        x = np.random.randn(nr_img, nr_in_channel, sz_img, 4)
        w = np.random.randn(nr_filter, nr_in_channel, sz_filter, sz_filter)
        # x = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        # w = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2]).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

        dout = np.random.randn(nr_img, nr_in_channel - 1, sz_img, 4) # standin for the derivitive from next layer

        # dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, conv_params)[0], x, dout)
        # dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, conv_params)[0], w, dout)

        y, cache = conv_forward(x, w, conv_param=conv_params)
        dx, dw = convolution_backward(dout, cache)

        # Your errors should be around e-8 or less.
        # print('Testing conv_backward_naive function')
        # print('dx error: ', rel_error(dx, dx_num))
        # print('dw error: ', rel_error(dw, dw_num))


    # def test_conv_backward_im2col(self):
        # conv_params = {
        #     'stride': 2,
        #     'pad': 1
        # }
        #
        # nr_img = 2;
        # sz_img = 4;
        # nr_in_channel = 3;
        # sz_filter = 4;
        # nr_filter = 3;
        #
        # x_size = nr_img * nr_in_channel * sz_img * sz_img
        # w_size = nr_filter * nr_in_channel * sz_filter * sz_filter
        #
        # x = np.linspace(-.1, .5, x_size).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        # w = np.linspace(-0.2, 0.3, w_size).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)
        #
        # y, cache = conv_forward(x, w, conv_param=conv_params)
        #
        # sz_out = int(1 + (sz_img + 2 * conv_params['pad'] - sz_filter) / conv_params['stride'])
        # shape_y = (nr_img, nr_filter, sz_out, sz_out) # 4x2x5x5
        #
        # x_size = nr_img * nr_in_channel * sz_img * sz_img
        # w_size = nr_filter * nr_in_channel * sz_filter * sz_filter
        #
        # # input for backward
        # dy = np.linspace(-0.1, 0.5).reshape(shape_y)
        # dx = np.linspace(-.1, .5, x_size).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        # dw = np.linspace(-0.2, 0.3, w_size).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

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
