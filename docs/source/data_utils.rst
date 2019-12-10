.. _awnn_data_utils:

Data loading
============

* Currently each batch will read to the memory first, saved in the "loader" in "get_train_bach_mt"
* When data layer is setup, the "DATA_PARTITIONED_N" layout is used.
* do_concurrent_read will copy each data portion concurrently to data layer out blob.
* TODO: The temp buffer can be avoided in future


