#!/usr/bin/env bash
# Train models for MNIST

echo "run setting MNIST MLPNet MSE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3
echo "run setting MNIST MLPProb BCE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3
echo "run setting MNIST MLPLinear MSE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3
echo "run setting MNIST MLPLinearProb BCE 5-8-9-3"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=5 pos_two=8 neg_one=9 neg_two=3

echo "run setting MNIST MLPNet MSE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7
echo "run setting MNIST MLPProb BCE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7
echo "run setting MNIST MLPLinear MSE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7
echo "run setting MNIST MLPLinearProb BCE 4-9-5-7"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=4 pos_two=9 neg_one=5 neg_two=7

echo "run setting MNIST MLPNet MSE 7-9-5-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4
echo "run setting MNIST MLPProb BCE 7-9-5-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4
echo "run setting MNIST MLPLinear MSE 7-9-5-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4
echo "run setting MNIST MLPLinearProb BCE 7-9-5-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=7 pos_two=9 neg_one=5 neg_two=4

echo "run setting MNIST MLPNet MSE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4
echo "run setting MNIST MLPProb BCE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4
echo "run setting MNIST MLPLinear MSE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4
echo "run setting MNIST MLPLinearProb BCE 5-9-8-4"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=5 pos_two=9 neg_one=8 neg_two=4

echo "run setting MNIST MLPNet MSE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9
echo "run setting MNIST MLPProb BCE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9
echo "run setting MNIST MLPLinear MSE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9
echo "run setting MNIST MLPLinearProb BCE 3-5-8-9"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=3 pos_two=5 neg_one=8 neg_two=9

echo "run setting MNIST MLPNet MSE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPNet loss_option=MSE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5
echo "run setting MNIST MLPProb BCE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPProb loss_option=BCE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5
echo "run setting MNIST MLPLinear MSE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinear loss_option=MSE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5
echo "run setting MNIST MLPLinearProb BCE 3-8-5-5"
CUDA_VISIBLE_DEVICES=0 python sources/train_models.py dataset=MNIST model_option=MLPLinearProb loss_option=BCE \
pos_one=3 pos_two=8 neg_one=5 neg_two=5