#!/usr/bin/env bash

# Train different architectures (CNN and ResNet) for MNIST

echo "run setting MNIST CNN_MNIST MSE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3
echo "run setting MNIST ResNet MSE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3

echo "run setting MNIST CNN_MNIST MSE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7
echo "run setting MNIST ResNet MSE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7

echo "run setting MNIST CNN_MNIST MSE 7-9-5-4"
CUDA_VISIBLE_DEVICES=1 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4
echo "run setting MNIST ResNet MSE 7-9-5-4"
CUDA_VISIBLE_DEVICES=1 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4

echo "run setting MNIST CNN_MNIST MSE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4
echo "run setting MNIST ResNet MSE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4

echo "run setting MNIST CNN_MNIST MSE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9
echo "run setting MNIST ResNet MSE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9

echo "run setting MNIST CNN_MNIST MSE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=CNN_MNIST loss_option=MSE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5
echo "run setting MNIST ResNet MSE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=ResNet loss_option=MSE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5
