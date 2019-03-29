/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#pragma once

#include "utils/list.h"
#include "awnn/tensor.h"
#include "awnn/memory.h"
#include "awnn/logging.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum{
  MAX_STR_LENGTH=81
};

typedef struct {
  //uint id_param;
  char name[MAX_STR_LENGTH];
  tensor_t data;
  tensor_t diff;

  struct list_head list;
} param_t;


static inline void net_attach_param(struct list_head *l_params, char* name, tensor_t data, tensor_t diff){
  param_t * p_param = (param_t *)mem_alloc(sizeof(param_t));
  strncpy(p_param->name, name, sizeof(name));
  p_param->data = data;
  p_param->diff = diff;
  init_list_head(&p_param->list);
  list_add_tail(&p_param->list, l_params);
  PINF("-- attaching %s", p_param->name);
}

// 
static inline void net_free_params(struct list_head *l_params){
  struct list_head *p_node, *tmp;
  PINF("Freeing all net params: \n{");
  list_for_each_safe(p_node, tmp,l_params){
    list_del(p_node);
    param_t *p_param = list_entry(p_node, typeof(*p_param), list);
    PINF("-- freeing %s", p_param->name);
    tensor_destroy(p_param->data); 
    tensor_destroy(p_param->diff); 
  }

  PINF("}/");
}


static inline void net_print_params(struct list_head *l_params){
  param_t *p_param;
  PINF("Dumping all net params: \n{");
  list_for_each_entry(p_param, l_params, list){
    PSTR("%s,", p_param->name);
  }

  PINF("}/");
}

#ifdef __cplusplus
}
#endif
