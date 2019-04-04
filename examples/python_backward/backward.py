import numpy as np


def tpose1230(X):
    original_x_shape = X.shape
    X = X.flatten()
    out_cpy = [None] * X.size

    src_idx = 0
    for i in range(original_x_shape[0]):
        for j in range(original_x_shape[1] * original_x_shape[2] * original_x_shape[3]):
            target_idx = (i + j * original_x_shape[0])
            out_cpy[target_idx] = X[src_idx]
            src_idx += 1

    return np.array(out_cpy).reshape(original_x_shape[1], original_x_shape[2], original_x_shape[3], original_x_shape[0])

def convolution_backward(dout, cache):
    """
    A fast implementation of the backward pass for a convolution layer
    based on im2col and col2im.
    """
    x, w, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    num_filters, _, filter_height, filter_width = w.shape

    # reshape the derivative from upper layer
    # 1 - num filters
    # then reshape on the number of filters and bring to 2D
    # transpose will take a while
    print("BEFORE TRANSPOSE")
    print(dout)
    tpose = dout.transpose(1, 2, 3, 0)
    print("AFTER TRANSPOSE")
    print(tpose)

    dout_reshaped = dout.reshape(num_filters, -1)

    # transpose makes the width of (x_cols) the filters becomes
    # Dim 1,
    # gives the dLoss/dw which == (dL/dy * dy/dw) since dy/dw == X
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    # deriv x dL/dx = dL/dy * dy/dx
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)

    # convert back to multidimensional
    # is gonna take a while
    dx = col2im(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                filter_height, filter_width, pad, stride)

    return dx, dw


"""
    def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int padding, int stride):

    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                        dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

"""


# re organizes flattened columns into shape of the original image
# and removing padding

# if padding .. create empty array without padding
# and map to unpadded array
def col2im(cols, N, C, H, W, field_height, field_width, padding, stride):

    # x = np.empty((N, C, H, W), dtype=cols.dtype)
    HH = int((H + 2 * padding - field_height) / stride + 1)
    WW = int((W + 2 * padding - field_width) / stride + 1)
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded



"""
@cython.boundscheck(False)
cdef int col2im_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]
"""

def col2im_inner(cols, x_padded,
                 N, C, H, W, HH, WW,
                 field_height, field_width, padding, stride):

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]
