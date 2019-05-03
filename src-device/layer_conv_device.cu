#include "awnn/common.h"

#include "awnndevice/cublas_wrappers.cuh"
#include "awnndevice/device_utils.cuh"
#include "awnndevice/layer_conv_device.cuh"

static int _blocks { 1 };
static int _threads { 1 };

int set_blocks(int x) {
  _blocks = x;
  return _blocks;
}
int set_threads(int x) {
  _threads = x;
  return _threads;
}

/*
 * In the above naive version for the CPU, we stride through the target one
 * by one and then index into the source using knowledge about our
 * transposition operation.
 *
 * Having already reduced the transposition from its original form to this
 * more memory oriented approach, we will further reduce the problem here.
 *
 * In the above, the recognition that created the above loop is that the we
 * need to stride by the original dim 3... which is what j * t.dim.dims[3],
 * but the source was offset by i which is the number of the new dimensions.
 * The j loop occurs t.dim.dims[0] * t.dim.dims[1] * t.dim.dims[2] times
 * inside of i
 *
 * A problem that remains here is that the target index was previously based
 * on a loop.  Now we need to base the target on some relationship to the
 * thread location in the grid
 *
 * We additionally need to base the source index [i + j * t.dim.dims[3]]
 * on some relationship to the thread
 *
 * gonna start by making the assumption that the target is the threadIdx.x.
 * potentially if we can derive the mapping from the target to the source,
 * we can do a grid stride loop here
 *
 * Since we are starting from the assumption that we are mapping from a
 * known target index to a source index, we can look at the data access
 * patters.  We can note that we are using a modulus and an integer divider
 * to do the offsets into the source array.  Then the only question that
 * remains is which dimension in the source array should be used
 * to determine what we mod and divide by.
 *
 * I call this "group_size"  This represents the significant points in the
 * src array where the dimensions cause use to need to offset the input.
 */
static __global__ void _do_tensor_make_transpose_3012_device(tensor_t d_transpose, tensor_t d_src) {

  #ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_transpose_3012_device\n", threadIdx.x);
    printf("src N=%u, C=%u, H=%u, W=%u, transpose N=%u, C=%u, H=%u, W=%u\n", d_transpose.dim.dims[0], d_transpose.dim.dims[1], d_transpose.dim.dims[2], d_transpose.dim.dims[3], d_src.dim.dims[0], d_src.dim.dims[1], d_src.dim.dims[2], d_src.dim.dims[3]);
  }
#endif

  uint n = d_capacity(d_src);
  uint group_size = d_src.dim.dims[0] * d_src.dim.dims[1] * d_src.dim.dims[2];
  uint stride = d_src.dim.dims[3];

  for (auto i : grid_stride_range(0u, n)) {
    uint src_idx = i / group_size + (i % group_size) * stride;
    d_transpose.data[i] = d_src.data[src_idx];
  }
}

/*
 * explore different block sizes here
 */
tensor_t tensor_make_transpose_3012_device(tensor_t t) {
  uint const transposed_shape[] = { t.dim.dims[3], t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] };

  tensor_t d_src = tensor_make_copy_h2d(t);
  tensor_t d_transposed = tensor_make_copy_h2d(t);
  tensor_reshape_(&d_transposed, transposed_shape, ARRAY_SIZE(transposed_shape));
  
  PINF("device code is called");
  _do_tensor_make_transpose_3012_device<<<_blocks, _threads>>>(d_transposed, d_src);

  tensor_t h_transposed = tensor_make(transposed_shape, ARRAY_SIZE(transposed_shape));
  tensor_copy_d2h(h_transposed, d_transposed);


  tensor_destroy_device(&d_src);
  tensor_destroy_device(&d_transposed);

  return h_transposed;
}


/**
 * TODO
 *
 * This function should just do the padding operation in parallel.  Although below in my notes of the
 * inner im2col, I speculate that this operation can be eliminated, I am still going to provide it in
 * the initial cuda implementation
 */
static __global__ void _do_tensor_make_padded_square_input_device(tensor_t d_padded, tensor_t d_src, uint p, T pad_val)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_padded_square_input_device\n", threadIdx.x);
  }
#endif

  uint C, H, W, HH, WW;
  C = d_src.dim.dims[1];
  H = d_src.dim.dims[2];
  W = d_src.dim.dims[3];
  HH = H + 2 * p;
  WW = W + 2 * p;

  uint n = d_capacity(d_padded);

  uint new_img_sz = d_padded.dim.dims[1] * d_padded.dim.dims[2] * d_padded.dim.dims[3];
  uint channel_sz = d_padded.dim.dims[2] * d_padded.dim.dims[3];

  for (auto iter : grid_stride_range(0u, n)) {
    uint i = iter / new_img_sz;        // i is the target image
    uint j = (iter / channel_sz) % C;  // j is the channel in the image
    uint k = (iter / WW) % HH;         // k is the row in the image
    uint l = (iter % WW);              // l is the col in the current image

    uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
    if (k < p) {
      d_padded.data[target_idx] = pad_val;
    } else if (k >= (H + p)) {
      d_padded.data[target_idx] = pad_val;
    } else if (l < p) {
      d_padded.data[target_idx] = pad_val;
    } else if (l >= (W + p)) {
      d_padded.data[target_idx] = pad_val;
    } else {
      uint src_idx = i * C * H * W + j * H * W + (k - p) * W + (l - p);
      d_padded.data[target_idx] = d_src.data[src_idx];
    }
  }
}

tensor_t tensor_make_padded_square_input_device(tensor_t h_t, uint p, T val) {

  uint padded_shape[] = { h_t.dim.dims[0], h_t.dim.dims[1], h_t.dim.dims[2] + 2 * p, h_t.dim.dims[3] + 2 * p };
  tensor_t d_padded = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_t d_src = tensor_make_copy_h2d(h_t);

  PINF("device code is called");

  _do_tensor_make_padded_square_input_device<<<_blocks, _threads>>>(d_padded, d_src, p, val);

  tensor_t h_padded = tensor_make(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_copy_d2h(h_padded, d_padded);

  tensor_destroy_device(&d_padded);
  tensor_destroy_device(&d_src);

  return h_padded;
}


/**
 * TODO
 *
 * The inner should take the x_padded tensor and spread it out over the cols array
 * based on the stride and filter size.  Since we are doing the convolution with the
 * gemm approach, this is a core function.
 *
 * It should generally to how the convolution would work except that it is just doing
 * a copy instead of doing multiplication.  My suspicion is that the bulk of the
 * work from the convolution could be done here, and I'm not currently sure why
 * we would see an improvement in speed from GEMM.  Additionally, note that this
 * function takes in the padded array.  The padded array is used to allow a normal
 * stride in the loops in this fucntino, but it seems like the padded array could be
 * eliminated and this function could manage the padding elements as part of its
 * looping structure.
 *
 * If two loops are created (one for core elements, and one for padding elements)
 * a conditional could be avoided.
 */
static __global__ void _do_im2col_inner_device_naive_thread_per_filter(
    tensor_t cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH,
    uint WW, uint filter_height, uint filter_width, uint padding, uint stride)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_im2col_inner_device\n", threadIdx.x);
  }
#endif

  uint cols_d1  = cols.dim.dims[1];
  uint img_sz   = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint chan_sz  = x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint row_sz   = x_padded.dim.dims[2];

  uint filters_per_channel  = HH * WW;
  uint filters_per_image    = C * filters_per_channel;
  uint total_filters        = N * filters_per_image;

  for (auto iter : grid_stride_range(0u, total_filters)) {

    uint n = iter / filters_per_image;  // ii is the target image
    uint c = (iter / filters_per_channel) % C;  // jj is the channel in the image
    uint j = (iter / WW) % HH;
    uint k = (iter % WW);

    for (uint f_row = 0; f_row < filter_height; ++f_row) {  // for each row of filter (relative row)
      for (uint f_col = 0; f_col < filter_width; ++f_col) {  // for each col of filter

        uint row = c * filter_width * filter_height + f_row * filter_width + f_col;
        uint col = j * WW * N + k * N + n;
        uint target_idx = row * cols_d1 + col;
        uint src_idx = (n * img_sz) + (c * chan_sz) + (stride * j + f_row) * row_sz + stride * k + f_col;

//        printf("t_row=%u, t_col=%u, t_idx=%u, target_idx=%u, src_idx=%u, val=%f, row=%u, col=%u\n", t_row, t_col, t_idx, target_idx, src_idx, cols.data[target_idx], row, col);
        cols.data[target_idx] = x_padded.data[src_idx];
      }
    }
  }
}


/**
 * This function does a transformation of the input image batch (which
 * already has padding) to a flattened and stretched out version which
 * is arranged in such a way that the filters in the convolution apply
 * to each row in the new 2D array.
 *
 * The purpose of this is so that we can do a multiplication by dot
 * product with the filters.
 *
 * The mapping is between a 4D object to a 2D object, where each row
 * represents 1 stretched out filter, and each col represents one of
 * the elements that filter will apply to.
 *
 * The mapping between the two is not trivial however due to a needing
 * to map the index in the GPU to a the particular 4D idx in a filter.
 * Since the filters overlap, and there is a stride to account for,
 * the mapping has a sort of jumping behavior as the filter windows
 * "butt" up against the walls.
 *
 * My solution here required converting the 1D global threadIdx to
 * the 4D index of a particular filter though a bunch of operations.
 * The operations may not seem straight forward but the general idea
 * is to find the particular filter using our target index, then
 * target specific elements in that filter using real offsets in the
 * original 4D domain.
 *
 * It could also be though of as a 6D to 1D transformation because
 * the number of applications of the filters in a specific channel
 * can be thought of as 2D, and as such replace the normal height and
 * width elements... then the final 2D comes from the height and width
 * of the filters themselves.
 *
 *
 * This is (should be, need to prove) a significant improvement over
 * the previous version of this kernel.  Instead of allocating one
 * thread per filter, I allocate 1 thread per element.  This means
 * a significantly higher degree of parallelism can be achieved.
 */
static __global__ void _do_im2col_inner_device_thread_per_element(
    tensor_t cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH,
    uint WW, uint filter_height, uint filter_width, uint padding, uint stride)
{

  uint cols_d_1 = cols.dim.dims[1];
  uint img_sz = C * x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint chan_sz = x_padded.dim.dims[2] * x_padded.dim.dims[3];
  uint row_sz = x_padded.dim.dims[2];

  uint filter_size = filter_height * filter_width;

  uint filters_per_channel = HH * WW;
  uint filters_per_image = C * filters_per_channel;
  uint total_filters = N * filters_per_image;
  uint total_elements = filter_size * total_filters;

#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_im2col_inner_device_thread_per_element\n", threadIdx.x);
    printf("filters_per_chan=%u, filters_per_img=%u, total_filters=%u\n", filters_per_channel, filters_per_image, total_filters);
  }
#endif

  for (auto iter : grid_stride_range(0u, total_elements)) {
    uint n = iter / (filters_per_image * filter_size);  // nn is the target image
    uint c = (iter / (filters_per_channel * filter_size)) % C;  // cc is the channel of the target filter

    // locate the window
    uint window_index_linear = iter / filter_size;
    uint window_index_r  = (window_index_linear / HH) % WW;
    uint windows_index_c = window_index_linear % WW;

    // row and column in a particular filter
    uint f_row = (iter / filter_width) % filter_width;
    uint f_col = iter % filter_width;

    // offset the filter col and row by the dimensions of the real image
    uint row = c * filter_width * filter_height + f_row * filter_width + f_col;
    uint col = window_index_r * WW * N + windows_index_c * N + n;

    // get src index and target idx
    uint src_idx = (n * img_sz) + (c * chan_sz) + (stride * window_index_r + f_row) * row_sz + stride * windows_index_c + f_col;
    uint target_idx = row * cols_d_1 + col;

//    printf("n=%u, c=%u, window_index_r=%u, windows_index_c=%u, window_idx_linear=%u, f_row=%u, f_col=%u, first_elem=%u, target_idx=%u, src_idx=%u, val=%f, row=%u, col=%u\n", n, c, window_index_r, windows_index_c, window_index_linear, f_row, f_col, first_elem, target_idx, src_idx, cols.data[target_idx], row, col);

    cols.data[target_idx] = x_padded.data[src_idx]; // do copy
  }
}

/**
 * im2col_inner_device is a setup function for the real call to
 * actually launch the kernel. For now, it will allocate and
 * de-allocate / transfer mem to and from the GPU. In the pure
 * GPU based forward, this function will not be called, but
 * rather the _do... function will be called directly.
 */
status_t im2col_inner_device(tensor_t cols, tensor_t x_padded, uint N,  uint C,  uint H,  uint W,  uint HH, uint WW, uint filter_height, uint filter_width, uint padding, uint stride)
{

  tensor_t d_cols       = tensor_make_copy_h2d(cols);
  tensor_t d_x_padded   = tensor_make_copy_h2d(x_padded);
  cudaDeviceSynchronize();
  // TODO: make it handler lager size

  PINF("device code is called");
  _do_im2col_inner_device_naive_thread_per_filter<<<_blocks, _threads>>>(d_cols, d_x_padded, N, C, H, W, HH, WW, filter_height, filter_width, padding, stride);

  tensor_copy_d2h(cols, d_cols);

  tensor_destroy_device(&d_cols);
  tensor_destroy_device(&d_x_padded);

  return S_OK;
}


/*
 * This function just sets up the im2col.
 */
tensor_t im2col_device(tensor_t const d_x, tensor_t const d_w, conv_param_t const hparams)
{
  uint N, C, H, W, filter_height, filter_width;
  int stride, pad_sz;

  N = d_x.dim.dims[0];
  C = d_x.dim.dims[1];
  H = d_x.dim.dims[2];
  W = d_x.dim.dims[3];

  filter_height = d_w.dim.dims[2];
  filter_width  = d_w.dim.dims[3];

  stride = hparams.stride;
  pad_sz = hparams.padding;

  // Check dimensions
  assert((W + 2 * pad_sz - filter_width) % stride == 0);
  assert((H + 2 * pad_sz - filter_height) % stride == 0);

  uint HH = (H + 2 * pad_sz - filter_height) / stride + 1; // total strides needed over rows
  uint WW = (W + 2 * pad_sz - filter_width) / stride + 1; // total strides needed over cols


  // TODO : look into not allocating here... maybe check bounds in the inner
  // TODO :   and simply replace with 0's
  uint padded_shape[] = { d_x.dim.dims[0], d_x.dim.dims[1], d_x.dim.dims[2] + 2 * pad_sz, d_x.dim.dims[3] + 2 * pad_sz };
  tensor_t d_x_padded = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));  // ALLOC

  /////////////////////////////////////////////////////////////////////////////
  _do_tensor_make_padded_square_input_device<<<_blocks, _threads>>>(d_x_padded, d_x, pad_sz, 0);  // 0 is pad value
  /////////////////////////////////////////////////////////////////////////////

  uint flattened_x_shape[] = {C * filter_height * filter_width, N * HH * WW};

  tensor_t d_flattened_x = tensor_make_zeros_device(flattened_x_shape, ARRAY_SIZE(flattened_x_shape)); // ALLOC

  /////////////////////////////////////////////////////////////////////////////
  _do_im2col_inner_device_naive_thread_per_filter<<<_blocks, _threads>>>(d_flattened_x, d_x_padded, N, C, H, W, HH, WW, filter_height, filter_width, pad_sz, stride);
  /////////////////////////////////////////////////////////////////////////////

  tensor_destroy_device(&d_x_padded);

  return d_flattened_x;
}


/*
 * The assumption I am making here is that the tensors and other elements
 * that are necessary are already allocated on the GPU.
 */
status_t convolution_forward_device(cublasHandle_t handle, tensor_t const d_x, tensor_t d_w, lcache_t* hcache, conv_param_t const hparams, tensor_t d_y)
{
  // 1. flatten the input into vectors which represent the filters
  tensor_t d_flattened_x = im2col_device(d_x, d_w, hparams); // 2 x ALLOC

  // 2. setup and apply filters
  uint const original_w_shape[] = { d_w.dim.dims[0], d_w.dim.dims[1], d_w.dim.dims[2], d_w.dim.dims[3] };
  uint const reshaped_w_shape[] = { d_w.dim.dims[0], d_w.dim.dims[1] * d_w.dim.dims[2] * d_w.dim.dims[3] };
  tensor_reshape_(&d_w, reshaped_w_shape, ARRAY_SIZE(reshaped_w_shape));

  // apply filters with gemm here !!!
  tensor_t d_out = cublas_gemm_launch(handle, d_w, d_flattened_x); // ALLOC

  // put d_w back to original shape
  tensor_reshape_(&d_w, original_w_shape, ARRAY_SIZE(original_w_shape));

  // reshape output back into tensor
  uint const out_shape_2[] = { original_w_shape[0], d_y.dim.dims[2], d_y.dim.dims[3], d_x.dim.dims[0] };
  tensor_reshape_(&d_out, out_shape_2, ARRAY_SIZE(out_shape_2));

  // 3. transpose output
  uint const transposed_shape[] = { d_out.dim.dims[3], d_out.dim.dims[0], d_out.dim.dims[1], d_out.dim.dims[2] };
  tensor_reshape_(&d_y, transposed_shape, ARRAY_SIZE(transposed_shape));

  //////////////////////////////////////////////////////////////////////
  _do_tensor_make_transpose_3012_device<<<_blocks, _threads>>>(d_y, d_out);
  //////////////////////////////////////////////////////////////////////

  // fill cache
  // NOTE, the order matters should be x, w, flattened_x
  if(hcache) {
    lcache_push(hcache, d_x);
    lcache_push(hcache, d_w);
    lcache_push(hcache, d_flattened_x);
  } else {
    tensor_destroy_device(&d_flattened_x);
  }

  tensor_destroy_device(&d_out);

  return S_OK;
}


// Call this function from the host ONLY
// cache will be filled with device mem
status_t convolution_forward_device_host_harness(cublasHandle_t handle,
                                                 tensor_t h_x, tensor_t h_w,
                                                 lcache_t* hcache,
                                                 conv_param_t hparams,
                                                 tensor_t h_y)
{
  tensor_t d_x = tensor_make_copy_h2d(h_x);
  tensor_t d_w = tensor_make_copy_h2d(h_w);
  tensor_t d_y = tensor_make_copy_h2d(h_y);

  convolution_forward_device(handle, d_x, d_w, hcache, hparams, d_y);

  tensor_copy_d2h(h_x, d_x);
  tensor_copy_d2h(h_w, d_w);
  tensor_copy_d2h(h_y, d_y);

  // d_x is cached so don't destroy
  // d_w is cached so don't destroy
  tensor_destroy_device(&d_y);

  return S_OK;
}


/**
 * TODO
 *
 * This function should just do the padding operation in parallel.  Although below in my notes of the
 * inner im2col, I speculate that this operation can be eliminated, I am still going to provide it in
 * the initial CUDA implementation
 */
static __global__ void _do_tensor_make_remove_padding_square_device(tensor_t d_padded, tensor_t d_src, uint p)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_remove_padding_square_device\n", threadIdx.x);
  }
#endif

  uint C, H, W, HH, WW;
  C = d_src.dim.dims[1];
  H = d_src.dim.dims[2];
  W = d_src.dim.dims[3];
  HH = H - 2 * p;
  WW = W - 2 * p;

  uint n = d_capacity(d_padded);

  uint new_img_sz = d_padded.dim.dims[1] * d_padded.dim.dims[2] * d_padded.dim.dims[3];
  uint channel_sz = d_padded.dim.dims[2] * d_padded.dim.dims[3];

  for (auto iter : grid_stride_range(0u, n)) {
    uint i = iter / new_img_sz;        // i is the target image
    uint j = (iter / channel_sz) % C;  // j is the channel in the image
    uint k = (iter / WW) % HH;         // k is the row in the image
    uint l = (iter % WW);              // l is the col in the current image

    uint target_idx = i * C * HH * WW + j * HH * WW + k * WW + l;
    uint src_idx = i * C * H * W + j * H * W + (k + p) * W + (l + p);
    d_padded.data[target_idx] = d_src.data[src_idx];
  }
}

tensor_t tensor_make_remove_padding_square_device(tensor_t t, uint p) {
  uint padded_shape[] = { t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] - 2 * p, t.dim.dims[3] - 2 * p };
  tensor_t d_out = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_t d_src = tensor_make_copy_h2d(t);

  PINF("device code is called");
  _do_tensor_make_remove_padding_square_device<<<_blocks, _threads>>>(d_out, d_src, p);

  tensor_t h_out = tensor_make(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_copy_d2h(h_out, d_out);

  tensor_destroy_device(&d_out);
  tensor_destroy_device(&d_src);

  return h_out;
}





/*
 * The purpose of this function is to take a flattened 2D matrix and
 * expand it back into the 4D space it originated in.
 *
 * The intuition comes from the fact that when we collapsed the 4D
 * space to 2D in the im2col operation, we were particularly interested
 * taking elements related to a single application of a filter and
 * spreading them out in a row of a 2D matrix.
 *
 * Now we are interested in taking those elements and putting them
 * back to where they came from.  This is a non-trivial operation
 * however due to the reality that we need to consider that when
 * we performed the im2col operation, we duplicate elements.  So
 * when we return those elements back, we have to recombine the
 * related elements.
 *
 * Additionally, the same considerations must be made regarding the
 * strategy we have to follow.  Since elements in the target are
 * repeatedly accessed when we recombine elements, this operation
 * should be a good target for usage of shared memory.  However
 * since the operation is so complex, I have chosen to approach
 * this with multiple phases, just like the im2col above.
 *
 * This particular version is the "most naive." Each thread is
 * responsible for a substantial amount of work.  Note that each
 * thread carries out all the work in the 2 inner for loops.
 */
static __global__ void _do_col2im_inner_device_thread_per_filter(
    tensor_t d_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH,
    uint WW, uint field_height, uint field_width, uint padding, uint stride)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_col2im_inner_device\n");
  }
#endif

  uint dx_col_d1  = d_cols.dim.dims[1];
  uint x_p_d1     = x_padded.dim.dims[1];
  uint x_p_d2     = x_padded.dim.dims[2];
  uint x_p_d3     = x_padded.dim.dims[3];

  for (auto iter : grid_stride_range(0u, N * C * field_width * field_height)) {

    uint i = iter / (C * field_height * field_width);
    uint c = (iter / (field_height * field_width)) % C;  // jj is the channel in the image
    uint fi = iter / (field_width) % field_height;
    uint fj = iter % field_width;

    uint row = c * field_width * field_height + fi * field_width + fj;

    for (uint h = 0; h < HH; ++h) {
      for (uint w = 0; w < WW; ++w) {
        uint col = h * WW * N + w * N + i;
        uint src_idx = row * dx_col_d1 + col;
        uint target_idx =
            i * x_p_d1 * x_p_d2 * x_p_d3
            + c * x_p_d2 * x_p_d3
            + (stride * h + fi) * x_p_d3
            + stride * w + fj;

#ifdef AWNN_USE_FLT32
          atomicAdd(&(x_padded.data[target_idx]), d_cols.data[src_idx]);
#else
          atomicAddDouble(&(x_padded.data[target_idx]), d_cols.data[src_idx]);
#endif
      }
    }
  }
}


static __global__ void _do_col2im_inner_device_thread_per_element(
    tensor_t d_cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH,
    uint WW, uint field_height, uint field_width, uint padding, uint stride)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    printf("entered _do_col2im_inner_device\n");
  }
#endif

  uint dx_col_d1  = d_cols.dim.dims[1];
  uint x_p_d1     = x_padded.dim.dims[1];
  uint x_p_d2     = x_padded.dim.dims[2];
  uint x_p_d3     = x_padded.dim.dims[3];

  uint C_fh_fw_HH_WW = C * field_height * field_width * HH * WW;

  for (auto iter : grid_stride_range(0u, N * C * H * W * field_height * field_width)) {

    uint i = iter / C_fh_fw_HH_WW;  // ii is the target image
    uint c = (iter / (field_height * field_width * HH * WW)) % C;  // jj is the channel in the image
    uint fi = iter / (HH * WW * field_width) % field_height;
    uint fj = (iter / (HH * WW)) % field_width;
    uint h = (iter / WW) % HH;
    uint w = iter % WW;

    uint row = c * field_width * field_height + fi * field_width + fj;

    uint col = h * WW * N + w * N + i;
    uint src_idx = row * dx_col_d1 + col;
    uint target_idx =
        i * x_p_d1 * x_p_d2 * x_p_d3
        + c * x_p_d2 * x_p_d3
        + (stride * h + fi) * x_p_d3
        + stride * w + fj;

#ifdef AWNN_USE_FLT32
    atomicAdd(&(x_padded.data[target_idx]), d_cols.data[src_idx]); // TODO fix this
#else
    atomicAddDouble(&(x_padded.data[target_idx]), d_cols.data[src_idx]);
#endif
  }
}

void col2im_inner_device(tensor_t cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW, uint field_height, uint field_width, uint padding, uint stride) {
  tensor_t d_cols       = tensor_make_copy_h2d(cols);
  tensor_t d_x_padded   = tensor_make_copy_h2d(x_padded);

  PINF("device code is called");
  _do_col2im_inner_device_thread_per_filter<<<_blocks, _threads>>>(d_cols, d_x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride);

  tensor_copy_d2h(x_padded, d_x_padded);

  tensor_destroy_device(&d_cols);
  tensor_destroy_device(&d_x_padded);
}


tensor_t col2im_device(tensor_t d_dx_cols, uint N, uint C, uint H, uint W, uint field_height, uint field_width, uint pad_sz, uint stride)
{
  uint HH = (H + 2 * pad_sz - field_height) / stride + 1;
  uint WW = (W + 2 * pad_sz - field_width) / stride + 1;

  uint x_padded_shape[] = { N, C, H + 2 * pad_sz, W + 2 * pad_sz };
  tensor_t d_x_padded = tensor_make_zeros_device(x_padded_shape, ARRAY_SIZE(x_padded_shape));  // new mem created by returned

  ////////////////////////////////////////////////////////////////////////////
  _do_col2im_inner_device_thread_per_filter<<<_blocks, _threads>>>(d_dx_cols, d_x_padded, N, C, H, W, HH, WW, field_height, field_width, pad_sz, stride);
  ////////////////////////////////////////////////////////////////////////////

  if (pad_sz) {
    uint padded_shape[] = { d_x_padded.dim.dims[0], d_x_padded.dim.dims[1], d_x_padded.dim.dims[2] - 2 * pad_sz, d_x_padded.dim.dims[3] - 2 * pad_sz };
    tensor_t padding_removed = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));
    ////////////////////////////////////////////////////////////////////////////
    _do_tensor_make_remove_padding_square_device<<<_blocks, _threads>>>(padding_removed, d_x_padded, pad_sz);
    ////////////////////////////////////////////////////////////////////////////

//    tensor_t padding_removed = tensor_make_remove_padding_square_device(d_x_padded, pad_sz);
    tensor_destroy_device(&d_x_padded);
    return padding_removed;
  }
  return d_x_padded;
}


static __global__ void _do_tensor_make_transpose_1230_device(tensor_t d_t, tensor_t d_src)
{
#ifdef DEBUG_GPU_FUNC_ENTRY_NOTIFY
  if (threadIdx.x == 0) {
    assert(d_src.mem_type == GPU_MEM);
    assert(d_t.mem_type == GPU_MEM);
    printf("entered _do_tensor_make_transpose_1230_device\n", threadIdx.x);
  }
#endif

  uint target_idx = 0;

  uint og_dim_1 = d_src.dim.dims[1];
  uint og_dim_2 = d_src.dim.dims[2];
  uint og_dim_3 = d_src.dim.dims[3];

  uint n = d_capacity(d_src);
  uint group_size = og_dim_1 * og_dim_2 * og_dim_3;
  uint stride = d_src.dim.dims[0];

  for (auto i : grid_stride_range(0u, n)) {
    target_idx = i / group_size + (i % group_size) * stride;
    d_t.data[target_idx] = d_src.data[i];
  }
}

tensor_t tensor_make_transpose_1230_device(tensor_t t)
{
  uint const transposed_shape[] = { t.dim.dims[1], t.dim.dims[2], t.dim.dims[3], t.dim.dims[0] };

  tensor_t d_src = tensor_make_copy_h2d(t);
  tensor_t d_transposed = tensor_make_copy_h2d(t);
  tensor_reshape_(&d_transposed, transposed_shape, ARRAY_SIZE(transposed_shape));

  PINF("device code is called");
  _do_tensor_make_transpose_1230_device<<<_blocks, _threads>>>(d_transposed, d_src);

  tensor_t h_transposed = tensor_make(transposed_shape, ARRAY_SIZE(transposed_shape));
  tensor_copy_d2h(h_transposed, d_transposed);

  tensor_destroy_device(&d_src);
  tensor_destroy_device(&d_transposed);

  return h_transposed;
}

status_t convolution_backward_device(cublasHandle_t handle, tensor_t d_dx, tensor_t d_dw, lcache_t* dcache, conv_param_t const params, tensor_t const d_dout)
{
  // NOTE : the order of pop matters, should be flattened_x, d_w, d_x (reverse of forward)
  tensor_t d_x_cols   = lcache_pop(dcache);
  tensor_t d_w        = lcache_pop(dcache);
  tensor_t d_x        = lcache_pop(dcache);

  assert(d_dx.mem_type == GPU_MEM);
  assert(d_dw.mem_type == GPU_MEM);
  assert(d_x_cols.mem_type == GPU_MEM);
  assert(d_w.mem_type == GPU_MEM);
  assert(d_x.mem_type == GPU_MEM);

  uint num_filters    = d_w.dim.dims[0];
  uint w_channels     = d_w.dim.dims[1];
  uint filter_height  = d_w.dim.dims[2];
  uint filter_width   = d_w.dim.dims[3];

  // 1. tensor transpose 1230 the dout (derivative of output layer)
  uint const d_dout_T_1230_shape[] = { d_dout.dim.dims[1], d_dout.dim.dims[2], d_dout.dim.dims[3], d_dout.dim.dims[0] };
  tensor_t d_dout_T_1230 = tensor_make_device(d_dout_T_1230_shape, ARRAY_SIZE(d_dout_T_1230_shape));
  _do_tensor_make_transpose_1230_device<<<_blocks, _threads>>>(d_dout_T_1230, d_dout);

  // 2. reshape the dout_T to a 2D shape by collapsing the last 3 dims
  uint d_dout_2d_shape[] = { num_filters, d_dout_T_1230.dim.dims[1] * d_dout_T_1230.dim.dims[2] * d_dout_T_1230.dim.dims[3] };
  tensor_reshape_(&d_dout_T_1230, d_dout_2d_shape, ARRAY_SIZE(d_dout_2d_shape));

  // 3. 2D transpose the previously flattened x_cols
  // TODO : eliminate this by merging it with the matrix multiply in step 4.
  tensor_t d_x_cols_T = cublas_transpose_launch(handle, d_x_cols);

  // 4. 2D GEMM multiply the dout_T by the flat d_x_cols_T
  uint mult_shape[] = { d_dout_T_1230.dim.dims[0], d_x_cols_T.dim.dims[1] };
  tensor_reshape_(&d_dw, mult_shape, ARRAY_SIZE(mult_shape));
  int ret = cublas_gemm_launch(handle, d_dout_T_1230, d_x_cols_T, d_dw);
  assert(ret == S_OK);

  // 5. reshape d_dw to same shape as cached d_w
  uint d_dw_shape[] = { num_filters, w_channels, filter_height, filter_width };
  tensor_reshape_(&d_dw, d_dw_shape, ARRAY_SIZE(d_dw_shape));

  // done getting d_dw (device derivative of w)

  // 6. now get d_dx in column form by multiplying the d_w_T with the d_out

  // 6a. get transpose of dw
  uint d_w_shape[] = { num_filters, w_channels * filter_height * filter_width };
  tensor_reshape_(&d_w, d_w_shape, ARRAY_SIZE(d_w_shape));
  tensor_t d_w_T = cublas_transpose_launch(handle, d_w);

  // 6b : gotta get dx : first we get it in flat form,
  tensor_t d_dx_cols = cublas_gemm_launch(handle, d_w_T, d_dout_T_1230);

  // 6c : then we convert it back to tensor form
  tensor_t t = col2im_device(d_dx_cols, d_x.dim.dims[0], d_x.dim.dims[1], d_x.dim.dims[2], d_x.dim.dims[3], filter_height, filter_width, params.padding, params.stride);

  // TODO : get rid of allocation of t by passing d_dx into col2im_device
  /////////////////////////////////////////////////////////////////////////////
  tensor_copy_d2d<<<_blocks, _threads>>>(d_dx, t);
  /////////////////////////////////////////////////////////////////////////////

  // cache
  tensor_destroy_device(&d_x_cols);
  tensor_destroy_device(&d_dout_T_1230);
  tensor_destroy_device(&d_x_cols_T);
  tensor_destroy_device(&d_w_T);
  tensor_destroy_device(&t);

  return S_OK;
}


// Call this function from the host ONLY
// cache is expected to be filled with device mem
status_t convolution_backward_device_host_harness(cublasHandle_t handle, tensor_t h_dx, tensor_t h_dw, lcache_t* hcache, conv_param_t const params, tensor_t const h_dout)
{
  tensor_t d_dx = tensor_make_copy_h2d(h_dx);
  tensor_t d_dw = tensor_make_copy_h2d(h_dw);
  tensor_t d_dout = tensor_make_copy_h2d(h_dout);

  convolution_backward_device(handle, d_dx, d_dw, hcache, params, d_dout);

  tensor_copy_d2h(h_dx, d_dx);
  tensor_copy_d2h(h_dw, d_dw);
  tensor_copy_d2h(h_dout, d_dout);

  tensor_destroy_device(&d_dx);
  tensor_destroy_device(&d_dw);
  tensor_destroy_device(&d_dout);

  return S_OK;
}


void elementwise_add_device_host_harness(tensor_t h_a, tensor_t h_b) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_add_inplace_device<<<_blocks, _threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void elementwise_mul_device_host_harness(tensor_t h_a, tensor_t h_b) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_b = tensor_make_copy_h2d(h_b);

  elementwise_mul_inplace_device<<<_blocks, _threads>>>(d_a, d_b);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_b);
}


void apply_mask_device_host_harness(tensor_t h_a, tensor_t h_mask) {
  tensor_t d_a = tensor_make_copy_h2d(h_a);
  tensor_t d_mask = tensor_make_copy_h2d(h_mask);

  apply_mask_device<<<_blocks, _threads>>>(d_a, d_mask);
  tensor_copy_d2h(h_a, d_a);

  tensor_destroy_device(&d_a);
  tensor_destroy_device(&d_mask);
}
