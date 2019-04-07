//
// Created by lifen on 4/5/19.
//
#include "gtest/gtest.h"
#include "utils/data_cifar.h"

namespace {

// The fixture for testing class Foo.
class DataCifarTest : public ::testing::Test {
 protected:
  static data_loader_t loader;
};

data_loader_t DataCifarTest::loader;

TEST_F(DataCifarTest, TestOpen) {
  status_t ret = cifar_open(&loader, CIFAR_PATH);
  EXPECT_EQ(S_OK, ret);
}

TEST_F(DataCifarTest, TestBatch) {
  tensor_t x;
  label_t *labels;
  uint batch_sz = 50;  // this can be 50 000/50 = 1000 batches
  uint batch_id = 0;

  uint nr_train_img = 50000;

  for (uint i = 0; i < nr_train_img / batch_sz; i++) {
    EXPECT_EQ(batch_sz,
              get_train_batch(&loader, &x, &labels, batch_id, batch_sz));
  }
}

TEST_F(DataCifarTest, TestClose) {
  status_t ret = cifar_close(&loader);
  EXPECT_EQ(S_OK, ret);
}

}  // namespace
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
