.. _awnn_memory:

Memory management
=================

There are three several types of memory used:

* layer learnable params(data and diff)
* layer output(data and diff)
* layer input(data and diff)
* layer cache(data only)

They are all attached in the model_t:

::

  struct list_head list_all_params[1]; // list of all learnable params
  struct list_head list_layer_out[1]; // list of output of each layer
  struct list_head list_layer_in[1]; // list of input of each layer
  struct list_head list_layer_cache[1]; // list of layer cache.

The struct list_head is a list structure derived from linux kernel list.h.
To use it we need to first call *init_list_head(head)*,
then for each nodes, init_list_head is also used to init, then the node can be
added to the list by *list_add* or *list_add_tail*


Allocation
----------

Currently the net knows about the maximum batch size, so all the sizes
of memory types above can be inferred.

* In the mlp_init function, all the *output*, *learnable params*  are directly
allocated and added to the net using *net_attach_param*
* *input* can be special, since the input of upper layer is the output of the
  bottom layer, we still use the *list* structure above to track all the data/diff
  of input, but they are just shadow copy of lower output.
* *cache* Currently, each layer can different types of caches. For the simplicity,
  lcache_t is just a place holder to track caches used by *each layer*.
  inside each lcache_t, tensors can be pushed in and popped out


Access
---------

After *net_attach_param*, each param can be accessed by

::

  tensor_t w = net_get_param(model->list_all_params, "fc1.weight")->data
  tensor_t dw = net_get_param(model->list_all_params, "fc1.weight")->diff

Destroy
----------

Only problem currently is destroy input. Currently the shadow copy from the output
is simply a copy of tensor_t, which means free from one side needs to explicitly
synced to the other side.

For now, I just free output tensors, i will revisit it in future.

