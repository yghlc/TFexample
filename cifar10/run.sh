#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=1

### show help function
#python cifar10_train.py --help

out_dir=/home/hlc/experiment/tf_tutorial

# training
#python cifar10_train.py --train_dir=${out_dir}/cifar10_train \
#    --log_frequency=100

#evl

python cifar10_eval.py --eval_dir=${out_dir}/cifar10_eval \
    --checkpoint_dir=${out_dir}/cifar10_train
