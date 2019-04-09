/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "utils/list.h"
#include "awnn/memory.h"

#include "test_util.h"
#include "gtest/gtest.h"
#include "utils/debug.h"

namespace {

// The fixture for testing class Foo.
class DtypeListTest : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  DtypeListTest() {
    // You can do set-up work for each test here.
  }

  ~DtypeListTest() override {
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

  // Objects declared here can be used by all tests.
};


TEST_F(DtypeListTest, TestList) {
  struct fake_struct{
    int id;
    char c;
    struct list_head list;
  };

  // an empty list
  struct list_head head;
  init_list_head(&head);

  // do nothing, since no elem is in the list
  struct list_head * p_node;
  list_for_each(p_node, &head){
    EXPECT_EQ(0,1);
  }


  struct fake_struct *p_struct;
  list_for_each_entry(p_struct, &head, list){
    EXPECT_EQ(0,1);
  }

  // insert some elems
  const int NUM_ELEM = 10000;
  for(int i = 0 ; i<NUM_ELEM; i++){
    struct fake_struct * p_elem = (struct fake_struct *)mem_alloc(sizeof(fake_struct));
    ASSERT_NE(p_elem, nullptr);
    init_list_head(&p_elem->list);
    p_elem->id = i;
    list_add(&p_elem->list, &head);
  }

  EXPECT_EQ(10000,list_get_count(&head));

  // check each value of list
  int cnt = 0;
  list_for_each_entry(p_struct, &head, list){
    EXPECT_EQ(p_struct->id,NUM_ELEM-1-cnt);
    cnt++;
  }

  // TODO delete all of them
  EXPECT_EQ(cnt, NUM_ELEM);
}




}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


