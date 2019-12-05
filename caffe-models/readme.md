Follow original dataset instruction

    cd $CAFFE_ROOT
    ./data/cifar10/get_cifar10.sh
    ./examples/cifar10/create_cifar10.sh

Run with
```
OPM_NUM_THREADS=12 build/tools/caffe time -model models/resnet_simple/resnet_cifar.prototxt -iterations 10
```

With nnpack-backend
```
build/tools/caffe time -model models/resnet_simple/resnet_cifar_nnpack.prototxt
```
