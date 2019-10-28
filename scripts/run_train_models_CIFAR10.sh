#!/usr/bin/env bash

# Train models for CIFAR10

echo "run setting CIFAR10 MLPNet MSE 1-3-2-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=1 pos_two=3 neg_one=2 neg_two=2
echo "run setting CIFAR10 MLPProb BCE 1-3-2-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=1 pos_two=3 neg_one=2 neg_two=2
echo "run setting CIFAR10 MLPLinear MSE 1-3-2-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=1 pos_two=3 neg_one=2 neg_two=2
echo "run setting CIFAR10 MLPLinearProb BCE 1-3-2-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=1 pos_two=3 neg_one=2 neg_two=2

echo "run setting CIFAR10 MLPNet MSE 1-7-3-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=1 pos_two=7 neg_one=3 neg_two=2
echo "run setting CIFAR10 MLPProb BCE 1-7-3-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=1 pos_two=7 neg_one=3 neg_two=2
echo "run setting CIFAR10 MLPLinear MSE 1-7-3-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=1 pos_two=7 neg_one=3 neg_two=2
echo "run setting CIFAR10 MLPLinearProb BCE 1-7-3-2"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=1 pos_two=7 neg_one=3 neg_two=2

echo "run setting CIFAR10 MLPNet MSE 2-3-1-1"
CUDA_VISIBLE_DEVICES=1 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=2 pos_two=3 neg_one=1 neg_two=1
echo "run setting CIFAR10 MLPProb BCE 2-3-1-1"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=2 pos_two=3 neg_one=1 neg_two=1
echo "run setting CIFAR10 MLPLinear MSE 2-3-1-1"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=2 pos_two=3 neg_one=1 neg_two=1
echo "run setting CIFAR10 MLPLinearProb BCE 2-3-1-1"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=2 pos_two=3 neg_one=1 neg_two=1

echo "run setting CIFAR10 MLPNet MSE 4-5-6-6"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=4 pos_two=5 neg_one=6 neg_two=6
echo "run setting CIFAR10 MLPProb BCE 4-5-6-6"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=4 pos_two=5 neg_one=6 neg_two=6
echo "run setting CIFAR10 MLPLinear MSE 4-5-6-6"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=4 pos_two=5 neg_one=6 neg_two=6
echo "run setting CIFAR10 MLPLinearProb BCE 4-5-6-6"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=4 pos_two=5 neg_one=6 neg_two=6

echo "run setting CIFAR10 MLPNet MSE 1-2-3-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=1 pos_two=2 neg_one=3 neg_two=3
echo "run setting CIFAR10 MLPProb BCE 1-2-3-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=1 pos_two=2 neg_one=3 neg_two=3
echo "run setting CIFAR10 MLPLinear MSE 1-2-3-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=1 pos_two=2 neg_one=3 neg_two=3
echo "run setting CIFAR10 MLPLinearProb BCE 1-2-3-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=1 pos_two=2 neg_one=3 neg_two=3

echo "run setting CIFAR10 MLPNet MSE 4-6-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPNet loss_option=MSE \
pos_one=4 pos_two=6 neg_one=5 neg_two=5
echo "run setting CIFAR10 MLPProb BCE 4-6-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPProb loss_option=BCE \
pos_one=4 pos_two=6 neg_one=5 neg_two=5
echo "run setting CIFAR10 MLPLinear MSE 4-6-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinear loss_option=MSE \
pos_one=4 pos_two=6 neg_one=5 neg_two=5
echo "run setting CIFAR10 MLPLinearProb BCE 4-6-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=CIFAR10 model_option=MLPLinearProb loss_option=BCE \
pos_one=4 pos_two=6 neg_one=5 neg_two=5