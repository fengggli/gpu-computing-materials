import numpy as np

def tpose3012(out):
    shape1 = out.shape
    out_cpy = []
    for i in range(shape1[3]):
        for j in range(shape1[0] * shape1[1] * shape1[2]):
            out_cpy.append(out.flatten()[i + j * shape1[3]])
    return np.array(out_cpy).reshape(out.shape[3], out.shape[0], out.shape[1], out.shape[2])

def conv_forward(x: np.array, w: np.array, conv_param: dict):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    # setup output buffer
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # actually create the storage and fill it with 0's
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1

    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)



    ############## 1. flatten the input into vectors which represent the filters
    x_cols = get_flattened_x(x, w, conv_param)

    # print()
    # print(x_cols.shape)
    # np.set_printoptions(precision=15)
    # print(list(x_cols.reshape(1, -1)))

    ############### 2. this is where the filters are actually applied
    # the -1 will make the shape into a single column vector of length w.shape[0]
    # this just puts all the weights into a vector.  Then we do a numpy "dot" on the
    # reshaped filters and the x which was placed into rows that were resized based
    # on the filter size.
    reshaped_w = w.reshape((w.shape[0], -1))
    res = w.reshape((w.shape[0], -1)).dot(x_cols)


    # print(res.flatten())

    ##### convert output to appropriate shape based on ???
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])

    posed = tpose3012(out)


    # print()
    # print("BEFORE 'TRANSPOSE'")
    # print(out)

    ############### 3.  transpose output (not sure what this is doing ??????)
    # below is the order of the dimensions
    out = out.transpose(3, 0, 1, 2)

    assert np.equal(posed.all(), out.all())
    # print(out == posed)
    # print()
    # print("AFTER 'TRANSPOSE'")
    # print(out)
    # print(out.flatten())

    ##### fill cache
    cache = (x, w, conv_param, x_cols)

    return out, cache


def get_flattened_x(x: np.array, w: np.array, conv_param: dict):

    ##### Create output storage (in our C code, this is done in the test)
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    ##### convert the image to the flattened form
    # note that the length of the rows will be the total size of a filter.
    # The number of columns will be num_channels * (total times the filter is
    # applied).  That is dependent on the pad, stride, and size of the filters,
    # which is why we send w.shape[2] and [3]
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    return x_cols


def im2col_cython(x, filter_height, filter_width, padding, stride):

    N, C, H, W = x.shape

    HH = int((H + 2 * padding - filter_height) / stride + 1)  # number of times filter will be applied in height dimension
    WW = int((W + 2 * padding - filter_width) / stride + 1)   # number of times filter will be applied in width dimension

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # print(x_padded)

    # when creating our columns array, we are sizing it based on
    # (Channels * filter size, number of images * total filter applications
    cols = np.zeros((C * filter_height * filter_width, N * HH * WW), dtype=x.dtype)  # field_height and width are filter

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW, filter_height, filter_width, padding, stride)

    return cols


def im2col_cython_inner(cols, x_padded,
                        N, C, H, W, HH, WW,
                        filter_height, filter_width, padding, stride):

    print(cols.shape)
    print(list(cols.flatten()))
    print(x_padded.shape)
    print(list(x_padded.flatten()))
    img_sz = C * x_padded.shape[2] * x_padded.shape[3]
    chan_sz = x_padded.shape[2] * x_padded.shape[3]
    row_sz = x_padded.shape[2]

    # print(x_padded.shape)
    for c in range(C):  # for each channel
        for yy in range(HH):  #
            for xx in range(WW):
                for ii in range(filter_height):
                    for jj in range(filter_width):
                        row = c * filter_width * filter_height + ii * filter_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            test_idx = (i * img_sz) + (c * chan_sz) + (stride * yy + ii) * row_sz + stride * xx + jj
                            # assert (x_padded[i, c, stride * yy + ii, stride * xx + jj], x_padded[test_idx])

                            x_p_val = x_padded[i, c, stride * yy + ii, stride * xx + jj]
                            target_val = cols[row, col]
                            target_idx = row * filter_height * filter_height + col
                            cols[row, col] = x_p_val
                        # print()
                        # print(cols)

    print(cols.shape)
    print(list(cols.flatten()))
