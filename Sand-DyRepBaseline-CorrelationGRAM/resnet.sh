#!/bin/bash
bash tools/dist_train.sh 8 configs/strategies/DyRep/resnet.yaml resnet50 --experiment imagenet_res50_dyrep_3 --data-path ~/../msalameh/datasets/ImageNet --dist-port 29500 --seed 3 --dyrep -j 8
