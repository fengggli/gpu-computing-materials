.. _awnn_blob:

Blob allocation
=========================

Recent work has been done to explore different level of parallelisms. they didn't consider "topology".

Blobs are allocated using a data_layout description and a topology description.

* For example a naive data_Layout description can be (DATA_PARTITIONED_N), which means the "N" dimension will be partitioned: "data".
* The topology description descripes how the processes/workers are in the system.

For each topology, i only need to specify a set of common functionalities:

Also: reduce time is too small for resnet: 
