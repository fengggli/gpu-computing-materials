/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#pragma once

#include "utils/list.h"
#include "awnn/tensor.h"
#include "awnn/layer.h"
#include "awnn/memory.h"
#include "awnn/logging.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  //uint id_param;
  char name[MAX_STR_LENGTH];
  tensor_t data;
  tensor_t diff;

  struct list_head list;
} param_t;

/* Attach preallocated tensor to net*/
static inline void net_attach_param(struct list_head *l_params, char* name, tensor_t data, tensor_t diff){
  param_t * p_param = (param_t *)mem_alloc(sizeof(param_t));
  strncpy(p_param->name, name, sizeof(name));
  p_param->data = data;
  p_param->diff = diff;
  init_list_head(&p_param->list);
  list_add_tail(&p_param->list, l_params);
  PINF("-- attaching %s", p_param->name);
}

/* Deallocated tensor from net, and free all of them*/
static inline void net_free_params(struct list_head *l_params){
  struct list_head *p_node, *tmp;
  PINF("Freeing all net params: \n{");
  list_for_each_safe(p_node, tmp,l_params){
    list_del(p_node);
    param_t *p_param = list_entry(p_node, typeof(*p_param), list);
    PINF("-- freeing %s", p_param->name);
    tensor_destroy(p_param->data); 
    tensor_destroy(p_param->diff); 
    mem_free(p_param);
  }
  PINF("}/");
}

/* print current param names*/
static inline void net_print_params(struct list_head const *l_params){
  param_t *p_param;
  PINF("Dumping all net params: \n{");
  list_for_each_entry(p_param, l_params, list){
    PSTR("%s,", p_param->name);
  }
  PINF("}/");
}

/* Get the entry of a specific param*/
static inline param_t *net_get_param(struct list_head const * l_params, char const*name){
  param_t *p_param;
  list_for_each_entry(p_param, l_params, list){
    if(strcmp(name, p_param->name) == 0) return p_param;
  }
  return NULL;
}

/* Attach cache placeholder to net*/
static inline void net_attach_cache(struct list_head *l_cache, char* name){
  lcache_t * p_cache = (lcache_t *)mem_alloc(sizeof(lcache_t)); // cache for this layer
  strncpy(p_cache->name, name, sizeof(name));
  p_cache->count = 0;
  init_list_head(&p_cache->list);
  list_add_tail(&p_cache->list, l_cache); // add to the net's global list
  PINF("-- attaching cache:  %s", p_cache->name);
}

static inline void net_free_cache(struct list_head *l_cache){
  struct list_head *p_node, *tmp;
  PINF("Freeing all caches: \n{");
  list_for_each_safe(p_node, tmp,l_cache){
    list_del(p_node);
    lcache_t *p_cache = list_entry(p_node, typeof(*p_cache), list);
    // tensor_destroy(p_param->data);  <- layer them self should delete them
    PINF("-- freeing cache: %s", p_cache->name);
    mem_free(p_cache);
  }
  PINF("}/");
}

#ifdef __cplusplus
}
#endif


