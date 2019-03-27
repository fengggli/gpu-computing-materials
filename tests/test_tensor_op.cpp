#include "awnn/tensor.h"
#include "gtest/gtest.h"
#include "test_util.h"

#include "awnn/common.h"

namespace {

// The fixture for testing class Foo.
class TensorOpTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  TensorOpTest() { // You can do set-up work for each test here.
  }

  ~TensorOpTest() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case for Foo.
  // static tensor_t t;
};

//
// tensor_t TensorOpTest::t;

// Tests that the Foo::Bar() method does Abc.
TEST_F(TensorOpTest, DotWrongInput) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 4};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));

  uint const shape3[] = {3, 4};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  EXPECT_TRUE(S_ERR == tensor_matmul(in1, in2, out));
}

TEST_F(TensorOpTest, Dot) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {3, 2};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 2};
  out = tensor_make_patterned(shape3, dim_of_shape(shape3));

  // int correct_result[] = {10, 13, 28, 40};

  EXPECT_EQ(S_OK, tensor_matmul(in1, in2, out));
  // tensor_dump(out);
  EXPECT_EQ(out.data[0], 10);
  EXPECT_EQ(out.data[1], 13);
  EXPECT_EQ(out.data[2], 28);
  EXPECT_EQ(out.data[3], 40);
}

TEST_F(TensorOpTest, PLUS) {
  tensor_t in1, in2, out;
  uint const shape1[] = {2, 3};
  in1 = tensor_make_patterned(shape1, dim_of_shape(shape1));
  // tensor_dump(in1);

  uint const shape2[] = {2, 3};
  in2 = tensor_make_patterned(shape2, dim_of_shape(shape2));
  // tensor_dump(in2);

  uint const shape3[] = {2, 3};
  out = tensor_make(shape3, dim_of_shape(shape3));

  EXPECT_EQ(S_OK, tensor_add_sameshape(in1, in2, out));
  EXPECT_EQ(out.data[0], 0);
  EXPECT_EQ(out.data[1], 2);
  EXPECT_EQ(out.data[2], 4);
  EXPECT_EQ(out.data[3], 6);
  EXPECT_EQ(out.data[4], 8);
  EXPECT_EQ(out.data[5], 10);
}

TEST_F(TensorOpTest, PLUS_INPLACE) {
  tensor_t from, to;
  uint const shape1[] = {2, 3};
  from = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 3};
  to = tensor_make_patterned(shape2, dim_of_shape(shape2));

  EXPECT_EQ(S_OK, tensor_elemwise_op_inplace(to, from, TENSOR_OP_ADD));

  EXPECT_EQ(to.data[0], 0);
  EXPECT_EQ(to.data[1], 2);
  EXPECT_EQ(to.data[2], 4);
  EXPECT_EQ(to.data[3], 6);
  EXPECT_EQ(to.data[4], 8);
  EXPECT_EQ(to.data[5], 10);
}

TEST_F(TensorOpTest, RESHAPE) {
  tensor_t t;
  uint const shape1[] = {2, 3, 4};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint const shape2[] = {2, 13}; // this shall gives a error
  uint const shape3[] = {2, 3, 2, 2};
  uint const shape4[] = {2, 12};
  uint const shape5[] = {24};

  EXPECT_EQ(S_BAD_DIM, tensor_reshape_(&t, shape2, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape3, 4));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape4, 2));
  EXPECT_EQ(S_OK, tensor_reshape_(&t, shape5, 1));

  EXPECT_EQ(t.dim.dims[0], 24);
  EXPECT_EQ(t.dim.dims[1], 0);
}


TEST_F(TensorOpTest, AddVector) {
  tensor_t t, v;
  uint const shape1[] = {2, 3, 4, 5};
  uint const shape2[] = {5};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));
  v = tensor_make_patterned(shape2, dim_of_shape(shape2));

  EXPECT_EQ(S_OK, tensor_add_vector_inplace(t, v));

  EXPECT_EQ(t.data[0], 0);
  EXPECT_EQ(t.data[1], 2);
  EXPECT_EQ(t.data[5], 5);

  tensor_destroy(t);
  tensor_destroy(v);
}

TEST_F(TensorOpTest, tensor_reshape_flat_) {
  tensor_t t;
  uint const shape1[] = {2, 3, 4, 5};
  t = tensor_make_patterned(shape1, dim_of_shape(shape1));

  uint original_capacity_of_t = tensor_get_capacity(t);

  tensor_reshape_flat_(&t);

  uint new_capacity_of_t = tensor_get_capacity(t);

  ASSERT_EQ(original_capacity_of_t, new_capacity_of_t);

  int i;
  for (i = 0; i < MAX_DIM - 1; ++i) {
    ASSERT_EQ(t.dim.dims[i], 1);
  }
  ASSERT_EQ(t.dim.dims[i], original_capacity_of_t);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test1) {
  uint const shape[] = { 1, 1, 1, 1 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test2) {
  uint const shape[] = { 1, 1, 1, 1 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);
//
//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test3) {
  uint const shape[] = { 1, 1, 1, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 1 x 1 -> 1 x 1 x 3 x 3
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test4) {
  uint const shape[] = { 1, 1, 2, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 2 x 2 -> 1 x 1 x 4 x 4
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test5) {
  uint const shape[] = { 1, 1, 2, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 2 x 2 -> 1 x 1 x 6 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test6) {
  uint const shape[] = { 1, 1, 3, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 5 x 4
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test7) {
  uint const shape[] = { 1, 1, 3, 2 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}


TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test8) {
  uint const shape[] = { 1, 1, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 2;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test9) {
  uint const shape[] = { 1, 2, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test10) {
  uint const shape[] = { 2, 1, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

TEST_F(TensorOpTest, tensor_make_padded_square_input_unit_test11) {
  uint const shape[] = { 2, 2, 2, 3 };
  tensor_t in = tensor_make_patterned(shape, dim_of_shape(shape));

  uint pad_size = 1;
  float pad_val = 0;

  // 1 x 1 x 3 x 2 -> 1 x 1 x 7 x 6
  tensor_t padded_in = tensor_make_padded_square_input(in, pad_size, pad_val);

//  tensor_dump(in);
//  tensor_dump(padded_in);

  ASSERT_EQ(in.dim.dims[2] + 2 * pad_size, padded_in.dim.dims[2]);
  ASSERT_EQ(in.dim.dims[3] + 2 * pad_size, padded_in.dim.dims[3]);

  uint h = padded_in.dim.dims[2];
  uint w = padded_in.dim.dims[3];
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      uint target_idx = i * w + j;
      if (i < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (i >= h - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j < pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else if (j >= w - pad_size) {
        ASSERT_EQ(pad_val, padded_in.data[target_idx]);
        int a = 0;
      } else {
        uint src_idx = (i - pad_size) * (w - 2 * pad_size) + j - pad_size;
        ASSERT_EQ(padded_in.data[target_idx], in.data[src_idx]);
      }
    }
  }
  tensor_destroy(in);
  tensor_destroy(padded_in);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
