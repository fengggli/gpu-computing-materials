#include "awnn/layer_conv.h"
#include "cuda_defs.h"
#include "range.cuh"


// type alias to simplify typing...
using namespace util::lang;
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
static __device__ step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

static uint __device__ capacity(tensor_t t) {
  return t.dim.dims[0] * t.dim.dims[1] * t.dim.dims[2] * t.dim.dims[3];
}

/*

tensor_t tensor_make_transpose_3012(tensor_t t) {
  uint target_idx = 0;
  tensor_t tpose = tensor_make_copy(t);

  for (uint i = 0; i < t.dim.dims[3]; ++i) {  // for each of the new dim 0
    for (uint j = 0; j < t.dim.dims[0] * t.dim.dims[1] * t.dim.dims[2]; ++j) {
      tpose.data[target_idx++] = t.data[i + j * t.dim.dims[3]];
    }
  }

  uint const shape[] = { t.dim.dims[3], t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] };
  tensor_reshape_(&tpose, shape, ARRAY_SIZE(shape));
  return tpose;
}
*/

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
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_transpose_3012_device\n", threadIdx.x);
    printf("src N=%u, C=%u, H=%u, W=%u, transpose N=%u, C=%u, H=%u, W=%u\n", d_transpose.dim.dims[0], d_transpose.dim.dims[1], d_transpose.dim.dims[2], d_transpose.dim.dims[3], d_src.dim.dims[0], d_src.dim.dims[1], d_src.dim.dims[2], d_src.dim.dims[3]);
  }

  uint n = capacity(d_src);
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

  dim3 threads(1024);
  dim3 blocks(1);
  PINF("device code is called");
  _do_tensor_make_transpose_3012_device<<<blocks, threads>>>(d_transposed, d_src);

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
 * the intitial cuda implementation
 */
static __global__ void _do_tensor_make_padded_square_input_device(tensor_t d_padded, tensor_t d_src, uint p, T pad_val)
{
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_padded_square_input_device\n", threadIdx.x);
  }

  uint C, H, W, HH, WW;
  C = d_src.dim.dims[1];
  H = d_src.dim.dims[2];
  W = d_src.dim.dims[3];
  HH = H + 2 * p;
  WW = W + 2 * p;

  uint n = capacity(d_padded);

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

tensor_t tensor_make_padded_square_input_device(tensor_t t, uint p, T val) {

  uint padded_shape[] = { t.dim.dims[0], t.dim.dims[1], t.dim.dims[2] + 2 * p, t.dim.dims[3] + 2 * p };
  tensor_t d_padded = tensor_make_device(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_t d_src = tensor_make_copy_h2d(t);

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_tensor_make_padded_square_input_device<<<blocks, threads>>>(d_padded, d_src, p, val);

  tensor_t h_padded = tensor_make(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_copy_d2h(h_padded, d_padded);

  tensor_destroy_device(&d_padded);
  tensor_destroy_device(&d_src);

  return h_padded;
}


static __global__ void _do_im2col_device(tensor_t const x, tensor_t const w, conv_param_t const params) {
  if (threadIdx.x == 0) {
    printf("entered _do_im2col_device\n", threadIdx.x);
  }
}

/*
 * This function just sets up the im2col.
 */
tensor_t im2col_device(tensor_t const x, tensor_t const w, conv_param_t const params)
{
  // TODO: make it handler lager size
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_im2col_device<<<blocks, threads>>>(x, w, params);
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
  if (threadIdx.x == 0) {
    printf("entered _do_im2col_inner_device\n", threadIdx.x);
  }
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

  if (threadIdx.x == 0) {
    printf("entered _do_im2col_inner_device_thread_per_element\n", threadIdx.x);
    printf("filters_per_chan=%u, filters_per_img=%u, total_filters=%u\n", filters_per_channel, filters_per_image, total_filters);
  }

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
 * im2col_inner_device is a setup function for the real call to actually launch the kernel.
 * For now, it will allocate and de-allocate / transfer mem to and from the GPU. In the pure
 * GPU based forward, this function will not be called, but rather the _do... function will be
 * called directly.
 */
status_t im2col_inner_device(tensor_t cols, tensor_t x_padded, uint N,  uint C,  uint H,  uint W,  uint HH, uint WW, uint filter_height, uint filter_width, uint padding, uint stride)
{

  tensor_t d_cols       = tensor_make_copy_h2d(cols);
  tensor_t d_x_padded   = tensor_make_copy_h2d(x_padded);

  // TODO: make it handler lager size
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_im2col_inner_device_thread_per_element<<<blocks, threads>>>(d_cols, d_x_padded, N, C, H, W, HH, WW, filter_height, filter_width, padding, stride);

  tensor_copy_d2h(cols, d_cols);

  tensor_destroy_device(&d_cols);
  tensor_destroy_device(&d_x_padded);

  return S_OK;
}


/*
 * Note that this is the only one that should likely remain global in the forward path.
 * The rest should become __device__ and should be called by this function
 */
static __global__ void _do_convolution_forward_device(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y)
{
  if (threadIdx.x == 0) {
    printf("entered _do_convolution_forward_device\n", threadIdx.x);
  }
}

/*
 * primary entry point for the forward function
 */
status_t convolution_forward_device(tensor_t const x, tensor_t const w, lcache_t* cache, conv_param_t const params, tensor_t y)
{

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");

  _do_convolution_forward_device<<<blocks, threads>>>(x, w, cache, params, y);
  return S_ERR;
}








static __global__ void _do_convolution_backward_device(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout)
{
  if (threadIdx.x == 0) {
    printf("entered _do_col2im_inner_device\n");
  }
}

status_t convolution_backward_device(tensor_t dx, tensor_t dw, lcache_t* cache, conv_param_t const params, tensor_t const dout)
{
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");
  _do_convolution_backward_device<<<blocks, threads>>>(dx, dw, cache, params, dout);
}


/**
 * TODO
 *
 * This function should just do the padding operation in parallel.  Although below in my notes of the
 * inner im2col, I speculate that this operation can be eliminated, I am still going to provide it in
 * the intitial cuda implementation
 */
static __global__ void _do_tensor_make_remove_padding_square_device(tensor_t d_padded, tensor_t d_src, uint p)
{
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_remove_padding_square_device\n", threadIdx.x);
  }
  uint C, H, W, HH, WW;
  C = d_src.dim.dims[1];
  H = d_src.dim.dims[2];
  W = d_src.dim.dims[3];
  HH = H - 2 * p;
  WW = W - 2 * p;

  uint n = capacity(d_padded);

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

  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");
  _do_tensor_make_remove_padding_square_device<<<blocks, threads>>>(d_out, d_src, p);

  tensor_t h_out = tensor_make(padded_shape, ARRAY_SIZE(padded_shape));
  tensor_copy_d2h(h_out, d_out);

  tensor_destroy_device(&d_out);
  tensor_destroy_device(&d_src);

  return h_out;
}


static __global__ void _do_col2im_device(tensor_t cols, uint N, uint C, uint H, uint W, uint field_height, uint field_width, uint padding, uint stride)
{
  if (threadIdx.x == 0) {
    printf("entered _do_col2im_inner_device\n");
  }
}

tensor_t col2im_device(tensor_t cols, uint N, uint C, uint H, uint W, uint field_height, uint field_width, uint padding, uint stride)
{
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");
  _do_col2im_device<<<blocks, threads>>>(cols, N, C, H, W, field_height, field_width, padding, stride);
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
static __global__ void _do_col2im_inner_device(tensor_t dx_cols, tensor_t d_x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW, uint field_height, uint field_width, uint padding, uint stride)
{
  if (threadIdx.x == 0) {
    printf("entered _do_col2im_inner_device\n");
  }

  uint dx_col_d1  = dx_cols.dim.dims[1];
  uint x_p_d1     = x_padded.dim.dims[1];
  uint x_p_d2     = x_padded.dim.dims[2];
  uint x_p_d3     = x_padded.dim.dims[3];

  uint C_fh_fw_HH_WW = C * field_height * field_width * HH * WW;

//  printf("\n(N=%u, C=%u, H=%u, W=%u, HH=%u, WW=%u, field_h=%u, field_w=%u, p=%u, stride=%u)\n", N, C, H, W, HH, WW, field_height, field_width, padding, stride);


  for (auto iter : grid_stride_range(0u, total_filters)) {
    uint row = c * field_width * field_height + fi * field_width + fj;

    uint i = iter / C_fh_fw_HH_WW;  // ii is the target image
    uint c = (iter / (field_height * field_width * HH * WW)) % C;  // jj is the channel in the image
    uint fi = iter / (HH * WW * field_width) % field_height;
    uint fj = (iter / (HH * WW)) % field_width;

    for (uint h = 0; h < HH; ++h) {
      for (uint w = 0; w < WW; ++w) {
//        printf("iter=%u, i=%u, c=%u, fi=%u, fj=%u, h=%u, w=%u\n", iter, i, c, fi, fj, h, w);
        uint col = h * WW * N + w * N + i;
        uint src_idx = row * dx_col_d1 + col;
        uint target_idx =
            i * x_p_d1 * x_p_d2 * x_p_d3
            + c * x_p_d2 * x_p_d3
            + (stride * h + fi) * x_p_d3
            + stride * w + fj;
        x_padded.data[target_idx] += dx_cols.data[src_idx];
      }

  }
}

void col2im_inner_device(tensor_t cols, tensor_t x_padded, uint N, uint C, uint H, uint W, uint HH, uint WW, uint field_height, uint field_width, uint padding, uint stride) {
  tensor_t d_cols       = tensor_make_copy_h2d(cols);
  tensor_t d_x_padded   = tensor_make_copy_h2d(x_padded);

  assert(padding * padding == cols.dim.dims[1]);

  // TODO: make it handler lager size
  dim3 threads(32);
  dim3 blocks(1);
  PINF("device code is called");
  _do_col2im_inner_device<<<blocks, threads>>>(d_cols, d_x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride);

  tensor_copy_d2h(cols, d_cols);

  tensor_destroy_device(&d_cols);
  tensor_destroy_device(&d_x_padded);
}


static __global__ void _do_tensor_make_transpose_1230_device(tensor_t d_t, tensor_t d_src)
{
  if (threadIdx.x == 0) {
    printf("entered _do_tensor_make_transpose_1230_device\n", threadIdx.x);
  }

  uint src_idx = 0, target_idx = 0;
  uint original_dim_0 = d_src.dim.dims[0];
  uint og_dim_1 = d_src.dim.dims[1];
  uint og_dim_2 = d_src.dim.dims[2];
  uint og_dim_3 = d_src.dim.dims[3];

  uint n = capacity(d_src);
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

  dim3 threads(1024);
  dim3 blocks(1);
  PINF("device code is called");
  _do_tensor_make_transpose_1230_device<<<blocks, threads>>>(d_transposed, d_src);

  tensor_t h_transposed = tensor_make(transposed_shape, ARRAY_SIZE(transposed_shape));
  tensor_copy_d2h(h_transposed, d_transposed);

  tensor_destroy_device(&d_src);
  tensor_destroy_device(&d_transposed);

  return h_transposed;
}