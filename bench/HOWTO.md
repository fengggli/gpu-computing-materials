## Benchmarks

1. resnet forward/backward passes, corresponding experiments in th:
(set the covolution method in *set_conv_method*)
```
./bench/bench-net-resnet 256 &> ../results/logs/2019-05-09-awnn-resnet-nnpack-bs256.log
```

#### Bench caffe

For both intel caffe and bvlc caffe:

...
cmake -DBLAS=mkl -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=on ..
...
