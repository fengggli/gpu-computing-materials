import numpy as np


def conv_forward_im2col(x, w, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """

    ##### Create output storage (in our C code, this is done in the test)
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # actually create the storage and fill it with 0's
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)


    ##### convert the image to the flattened form
    # note that the length of the rows will be the total size of a filter.
    # The number of columns will be num_chanels * (total times the filter is
    # applied).  That is dependent on the pad, stride, and size of the filters,
    # which is why we send w.shape[2] and [3]

    # NOTE : this operation requires extra memory, and additionally
    #        walks over each chunk of memory the size of a filter, just
    #        to copy.  I'm currently not clear why the convolution would
    #        be faster this way than performing it directly.  Just look
    #        at the massive function in inner im2col, which loops over
    #        a ton of stuff in a very memory inefficient way.
    #        I would like to explore doing this mapping in CUDA during
    #        the transfer.
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)


    ##### this is where the filters are actually applied

    # the -1 will make the shape into a single column vector of length w.shape[0]
    # this just puts all the weights into a vector.  Then we do a numpy "dot" on the
    # reshaped filters and the x which was placed into rows that were resized based
    # on the filter size.
    res = w.reshape((w.shape[0], -1)).dot(x_cols)

    ##### convert output back to appropriate shape
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])

    ##### transpose output (not sure what this is doing ??????)
    out = out.transpose(3, 0, 1, 2)

    ##### fill cache
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



