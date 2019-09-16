/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#include "awnn/tensor.h"
#include "utils/list.h"

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_CACHE_SIZE (10)
/*
 * @brief layer cache
 *
 * Currently each forward will allocate its own cache, and used by its backward.
 * lcache_t will be created outside of the forward function(using make_lcache);
 * but the count and all_tensors will be populated inside the forward.
 * The reason is that different layers require different number of caches
 *
 * */
typedef struct {
  uint count;  // number of tensors
  tensor_t all_tensors[MAX_CACHE_SIZE];
  char name[MAX_STR_LENGTH];
  struct list_head list;  // for inter-layer traversal
} lcache_t;
void lcache_push(lcache_t *cache, tensor_t t);
tensor_t lcache_pop(lcache_t *cache);
void make_empty_lcache(lcache_t *cache);
void lcache_free_all(lcache_t *cache);
void lcache_dump_stat(lcache_t *cache);

#ifdef __cplusplus
}
#endif
