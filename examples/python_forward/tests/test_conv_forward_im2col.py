from unittest import TestCase
import numpy as np

from python_forward.forward import conv_forward_im2col

class TestConvForwardIm2col(TestCase):
    def test_conv_forward_im2col(self):
        conv_params = {
            'stride': 2,
            'pad': 1
        }

        nr_img = 2;
        sz_img = 4;
        nr_in_channel = 3;
        sz_filter = 4;
        nr_filter = 3;

        x_size = nr_img * nr_in_channel * sz_img * sz_img
        w_size = nr_filter * nr_in_channel * sz_filter * sz_filter

        x = np.linspace(-.1, .5, x_size).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        w = np.linspace(-0.2, 0.3, w_size).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

        y, cache = conv_forward_im2col(x, w, conv_param=conv_params)



    def test_conv_forward_from_picture(self):
        """
            this is the example from the picture we looked at... source is here
            https://www.microsoft.com/en-us/research/uploads/prod/2018/05/spg-cnn-asplos17.pdf
        """

        # conv_params = {
        #     'stride': 1,
        #     'pad': 0
        # }
        #
        # nr_img = 1
        # sz_img = 3
        # nr_in_channel = 2  # input channels
        # sz_filter = 2
        # nr_filter = 2  # num output channels
        #
        # x_size = nr_img * nr_in_channel * sz_img * sz_img
        # w_size = nr_filter * nr_in_channel * sz_filter * sz_filter
        #
        # x = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        # w = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2]).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)
        #
        # y, cache = conv_forward_im2col(x, w, conv_param=conv_params)
        # print()
        # print(y)
        # print(cache)
