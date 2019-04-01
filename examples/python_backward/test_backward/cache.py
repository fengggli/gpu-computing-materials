cache = tuple(
    [[[[-0.1       , -0.09368421, -0.08736842, -0.08105263],
         [-0.07473684, -0.06842105, -0.06210526, -0.05578947],
         [-0.04947368, -0.04315789, -0.03684211, -0.03052632],
         [-0.02421053, -0.01789474, -0.01157895, -0.00526316]],

        [[ 0.00105263,  0.00736842,  0.01368421,  0.02      ],
         [ 0.02631579,  0.03263158,  0.03894737,  0.04526316],
         [ 0.05157895,  0.05789474,  0.06421053,  0.07052632],
         [ 0.07684211,  0.08315789,  0.08947368,  0.09578947]],

        [[ 0.10210526,  0.10842105,  0.11473684,  0.12105263],
         [ 0.12736842,  0.13368421,  0.14      ,  0.14631579],
         [ 0.15263158,  0.15894737,  0.16526316,  0.17157895],
         [ 0.17789474,  0.18421053,  0.19052632,  0.19684211]]],


       [[[ 0.20315789,  0.20947368,  0.21578947,  0.22210526],
         [ 0.22842105,  0.23473684,  0.24105263,  0.24736842],
         [ 0.25368421,  0.26      ,  0.26631579,  0.27263158],
         [ 0.27894737,  0.28526316,  0.29157895,  0.29789474]],

        [[ 0.30421053,  0.31052632,  0.31684211,  0.32315789],
         [ 0.32947368,  0.33578947,  0.34210526,  0.34842105],
         [ 0.35473684,  0.36105263,  0.36736842,  0.37368421],
         [ 0.38      ,  0.38631579,  0.39263158,  0.39894737]],

        [[ 0.40526316,  0.41157895,  0.41789474,  0.42421053],
         [ 0.43052632,  0.43684211,  0.44315789,  0.44947368],
         [ 0.45578947,  0.46210526,  0.46842105,  0.47473684],
         [ 0.48105263,  0.48736842,  0.49368421,  0.5       ]]]],
    [[[[-0.2       , -0.1965035 , -0.19300699, -0.18951049],
         [-0.18601399, -0.18251748, -0.17902098, -0.17552448],
         [-0.17202797, -0.16853147, -0.16503497, -0.16153846],
         [-0.15804196, -0.15454545, -0.15104895, -0.14755245]],

        [[-0.14405594, -0.14055944, -0.13706294, -0.13356643],
         [-0.13006993, -0.12657343, -0.12307692, -0.11958042],
         [-0.11608392, -0.11258741, -0.10909091, -0.10559441],
         [-0.1020979 , -0.0986014 , -0.0951049 , -0.09160839]],

        [[-0.08811189, -0.08461538, -0.08111888, -0.07762238],
         [-0.07412587, -0.07062937, -0.06713287, -0.06363636],
         [-0.06013986, -0.05664336, -0.05314685, -0.04965035],
         [-0.04615385, -0.04265734, -0.03916084, -0.03566434]]],


       [[[-0.03216783, -0.02867133, -0.02517483, -0.02167832],
         [-0.01818182, -0.01468531, -0.01118881, -0.00769231],
         [-0.0041958 , -0.0006993 ,  0.0027972 ,  0.00629371],
         [ 0.00979021,  0.01328671,  0.01678322,  0.02027972]],

        [[ 0.02377622,  0.02727273,  0.03076923,  0.03426573],
         [ 0.03776224,  0.04125874,  0.04475524,  0.04825175],
         [ 0.05174825,  0.05524476,  0.05874126,  0.06223776],
         [ 0.06573427,  0.06923077,  0.07272727,  0.07622378]],

        [[ 0.07972028,  0.08321678,  0.08671329,  0.09020979],
         [ 0.09370629,  0.0972028 ,  0.1006993 ,  0.1041958 ],
         [ 0.10769231,  0.11118881,  0.11468531,  0.11818182],
         [ 0.12167832,  0.12517483,  0.12867133,  0.13216783]]],


       [[[ 0.13566434,  0.13916084,  0.14265734,  0.14615385],
         [ 0.14965035,  0.15314685,  0.15664336,  0.16013986],
         [ 0.16363636,  0.16713287,  0.17062937,  0.17412587],
         [ 0.17762238,  0.18111888,  0.18461538,  0.18811189]],

        [[ 0.19160839,  0.1951049 ,  0.1986014 ,  0.2020979 ],
         [ 0.20559441,  0.20909091,  0.21258741,  0.21608392],
         [ 0.21958042,  0.22307692,  0.22657343,  0.23006993],
         [ 0.23356643,  0.23706294,  0.24055944,  0.24405594]],

        [[ 0.24755245,  0.25104895,  0.25454545,  0.25804196],
         [ 0.26153846,  0.26503497,  0.26853147,  0.27202797],
         [ 0.27552448,  0.27902098,  0.28251748,  0.28601399],
         [ 0.28951049,  0.29300699,  0.2965035 ,  0.3       ]]]],
    {'stride': 2, 'pad': 1},
    [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        , -0.06842105,  0.23473684],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.07473684,
         0.22842105, -0.06210526,  0.24105263],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.06842105,
         0.23473684, -0.05578947,  0.24736842],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.06210526,
         0.24105263,  0.        ,  0.        ],
       [ 0.        ,  0.        , -0.09368421,  0.20947368,  0.        ,
         0.        , -0.04315789,  0.26      ],
       [-0.1       ,  0.20315789, -0.08736842,  0.21578947, -0.04947368,
         0.25368421, -0.03684211,  0.26631579],
       [-0.09368421,  0.20947368, -0.08105263,  0.22210526, -0.04315789,
         0.26      , -0.03052632,  0.27263158],
       [-0.08736842,  0.21578947,  0.        ,  0.        , -0.03684211,
         0.26631579,  0.        ,  0.        ],
       [ 0.        ,  0.        , -0.06842105,  0.23473684,  0.        ,
         0.        , -0.01789474,  0.28526316],
       [-0.07473684,  0.22842105, -0.06210526,  0.24105263, -0.02421053,
         0.27894737, -0.01157895,  0.29157895],
       [-0.06842105,  0.23473684, -0.05578947,  0.24736842, -0.01789474,
         0.28526316, -0.00526316,  0.29789474],
       [-0.06210526,  0.24105263,  0.        ,  0.        , -0.01157895,
         0.29157895,  0.        ,  0.        ],
       [ 0.        ,  0.        , -0.04315789,  0.26      ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [-0.04947368,  0.25368421, -0.03684211,  0.26631579,  0.        ,
         0.        ,  0.        ,  0.        ],
       [-0.04315789,  0.26      , -0.03052632,  0.27263158,  0.        ,
         0.        ,  0.        ,  0.        ],
       [-0.03684211,  0.26631579,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03263158,  0.33578947],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.02631579,
         0.32947368,  0.03894737,  0.34210526],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.03263158,
         0.33578947,  0.04526316,  0.34842105],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.03894737,
         0.34210526,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.00736842,  0.31052632,  0.        ,
         0.        ,  0.05789474,  0.36105263],
       [ 0.00105263,  0.30421053,  0.01368421,  0.31684211,  0.05157895,
         0.35473684,  0.06421053,  0.36736842],
       [ 0.00736842,  0.31052632,  0.02      ,  0.32315789,  0.05789474,
         0.36105263,  0.07052632,  0.37368421],
       [ 0.01368421,  0.31684211,  0.        ,  0.        ,  0.06421053,
         0.36736842,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.03263158,  0.33578947,  0.        ,
         0.        ,  0.08315789,  0.38631579],
       [ 0.02631579,  0.32947368,  0.03894737,  0.34210526,  0.07684211,
         0.38      ,  0.08947368,  0.39263158],
       [ 0.03263158,  0.33578947,  0.04526316,  0.34842105,  0.08315789,
         0.38631579,  0.09578947,  0.39894737],
       [ 0.03894737,  0.34210526,  0.        ,  0.        ,  0.08947368,
         0.39263158,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.05789474,  0.36105263,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.05157895,  0.35473684,  0.06421053,  0.36736842,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.05789474,  0.36105263,  0.07052632,  0.37368421,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.06421053,  0.36736842,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.13368421,  0.43684211],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.12736842,
         0.43052632,  0.14      ,  0.44315789],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.13368421,
         0.43684211,  0.14631579,  0.44947368],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.14      ,
         0.44315789,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.10842105,  0.41157895,  0.        ,
         0.        ,  0.15894737,  0.46210526],
       [ 0.10210526,  0.40526316,  0.11473684,  0.41789474,  0.15263158,
         0.45578947,  0.16526316,  0.46842105],
       [ 0.10842105,  0.41157895,  0.12105263,  0.42421053,  0.15894737,
         0.46210526,  0.17157895,  0.47473684],
       [ 0.11473684,  0.41789474,  0.        ,  0.        ,  0.16526316,
         0.46842105,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.13368421,  0.43684211,  0.        ,
         0.        ,  0.18421053,  0.48736842],
       [ 0.12736842,  0.43052632,  0.14      ,  0.44315789,  0.17789474,
         0.48105263,  0.19052632,  0.49368421],
       [ 0.13368421,  0.43684211,  0.14631579,  0.44947368,  0.18421053,
         0.48736842,  0.19684211,  0.5       ],
       [ 0.14      ,  0.44315789,  0.        ,  0.        ,  0.19052632,
         0.49368421,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.15894737,  0.46210526,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.15263158,  0.45578947,  0.16526316,  0.46842105,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.15894737,  0.46210526,  0.17157895,  0.47473684,  0.        ,
         0.        ,  0.        ,  0.        ],
       [ 0.16526316,  0.46842105,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ]])