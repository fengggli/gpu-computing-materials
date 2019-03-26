import numpy as np


def conv_forward_im2col(x, w, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)

    # print()
    # print(x)
    # print(x_cols)

    res = w.reshape((w.shape[0], -1)).dot(x_cols)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, conv_param, x_cols)
    return out, cache


def im2col_cython(x, field_height, field_width, padding, stride):

    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    HH = int((H + 2 * padding - field_height) / stride + 1)
    WW = int((W + 2 * padding - field_width) / stride + 1)

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
    cols = np.zeros((C * field_height * field_width, N * HH * WW), dtype=x.dtype)
    
    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.
    
    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    return cols


def im2col_cython_inner(cols, x_padded,
                        N,  C,  H,  W,  HH,  WW,
                        field_height, field_width, padding, stride):

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]



