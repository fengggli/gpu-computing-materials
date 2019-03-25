from unittest import TestCase
import numpy as np

from forward import conv_forward_im2col


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

        conv_forward_im2col(x, w, conv_param=conv_params)
