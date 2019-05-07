//
// Created by cmgoebel on 5/6/19.
//

#pragma once

#include "awnn/tensor.h"    // tensor_t
#include "awnn/layer.h"     // lcache_t

status_t global_avg_pool_forward_device(tensor_t const x, lcache_t *cache, tensor_t y);
status_t global_avg_pool_backward_device(tensor_t dx, lcache_t *cache, tensor_t const dy);
