from unittest import TestCase
import numpy as np

from python_forward.forward import conv_forward, tpose3012

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

        y, cache = conv_forward(x, w, conv_param=conv_params)



    def test_conv_forward_from_picture(self):
        """
            this is the example from the picture we looked at... source is here
            https://www.microsoft.com/en-us/research/uploads/prod/2018/05/spg-cnn-asplos17.pdf
        """

        conv_params = {
            'stride': 1,
            'pad': 0
        }

        nr_img = 1
        sz_img = 3
        nr_in_channel = 2  # input channels
        sz_filter = 2
        nr_filter = 2  # num output channels

        x = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(nr_img, nr_in_channel, sz_img, sz_img)
        w = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2]).reshape(nr_filter, nr_in_channel, sz_filter, sz_filter)

        y, cache = conv_forward(x, w, conv_param=conv_params)
        # print()
        # print(y)
        # print(cache)

    def test_tpose3012(self):
        """
        this function checks the manual transpose function that reshapes to a 3012
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

        p = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2]).reshape(1, 2, 3, 3)
        x = np.linspace(-.1, .5, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        w = np.linspace(-0.2, 0.3, 3 * 3 * 4 * 6).reshape(3, 3, 4, 6)

        self.assertTrue(np.array_equal(tpose3012(p), p.transpose(3, 0, 1, 2)))
        self.assertTrue(np.array_equal(tpose3012(w), w.transpose(3, 0, 1, 2)))
        self.assertTrue(np.array_equal(tpose3012(x), x.transpose(3, 0, 1, 2)))

        # print(list(p.transpose(3, 0, 1, 2).flatten()))
        # print(list(x.transpose(3, 0, 1, 2).flatten()))
        # print(list(w.transpose(3, 0, 1, 2).flatten()))

