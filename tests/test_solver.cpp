/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "awnn/solver.h"
#include "gtest/gtest.h"
#include "test_util.h"

namespace {

// The fixture for testing class Foo.
class SolverTest : public ::testing::Test {
 protected:
  static solver_handle_t solver;
};

solver_handle_t SolverTest:: solver;

// input of forward

TEST_F(SolverTest, Construct) {
  // input to the model
  data_t data;
  model_t model;

  // config for the mlp
  uint batch_sz = 50;
  uint input_dim = 5;
  uint output_dim = 7;
  uint nr_hidden_layers = 1;
  uint hidden_dims[] = {50};
  T reg = 0;

  solver_config_t solver_config;
  solver_config.batch_size = batch_sz;
  solver_config.optimize_method = OPTIM_SGD;
  solver_config.learning_rate = 0.1;

  mlp_init(&model, batch_sz, input_dim, output_dim, nr_hidden_layers,
           hidden_dims, reg);
  EXPECT_NE((void *)0,
            (void *)net_get_param(model.list_all_params, "fc0.weight"));

  EXPECT_EQ(S_OK, solver_init(&solver, &model, &data, &solver_config));
}

// update the model with one step
TEST_F(SolverTest, Update) {
  // input to the model
  solver_train(&solver);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
