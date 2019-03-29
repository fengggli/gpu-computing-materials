/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// List implementation following the /include/linux/list.h
// https://notes.shichao.io/lkd/ch6/
struct list_head{
  struct list_head *next;
  struct list_head *prev;
};

#define container_of(ptr, type, member) ({ \
	const typeof( ((type *)0)->member ) *__mptr = (ptr); \
  (type *)( (char *)__mptr - offsetof(type,member) );})

/*
 * get the struct of this entry
 * @param ptr struct list_head pointer
 * @param type  the type of struct this is embeded in.
 * @parm name of list_struct within the struct.
 */
#define list_entry(ptr, type, member) \
	container_of(ptr, type, member)


// init a circular doublely-linked list
static inline void init_list_head(struct list_head* list){
  list->next = list;
  list->prev = list;
}

static inline void _list_add(struct list_head *node, struct list_head *prev, struct list_head *next){
  next->prev = node;
  node->next = next;
  node->prev = prev;
  prev->next = node;
}


// insert node to the beginning of the list
static inline void list_add(struct list_head *node, struct list_head *head){
  _list_add(node, head, head->next);
}

static inline void _list_del(struct list_head* prev, struct list_head * next){
  next->prev = prev;
  prev->next = next;
}

static inline  void list_del(struct list_head * entry){
  _list_del(entry->prev, entry->next);
}

static inline int list_empty(const struct list_head *head){
  return (head->next == head);
}

#define list_for_each(pos, head) \
  for (pos = (head)->next; pos!=(head); pos = pos->next)

#define list_for_each_entry(pos, head, member) \
	for (pos = list_entry((head)->next, typeof(*pos), member);	\
		&pos->member != (head); 	\
		pos = list_entry(pos->member.next, typeof(*pos), member))


#ifdef __cplusplus
}
#endif
