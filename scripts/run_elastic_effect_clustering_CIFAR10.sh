#!/usr/bin/env bash
# Run local elasticity based clustering on CIFAR10

echo "run setting CIFAR10 ripple MLPNet MSE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 1-3-2-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=3 neg_one=2 neg_two=2 theta_option=optimal

echo "run setting CIFAR10 ripple MLPNet MSE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 1-7-3-2 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=7 neg_one=3 neg_two=2 theta_option=optimal

echo "run setting CIFAR10 ripple MLPNet MSE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 2-3-1-1 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=2 pos_two=3 neg_one=1 neg_two=1 theta_option=optimal

echo "run setting CIFAR10 ripple MLPNet MSE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 4-5-6-6 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=5 neg_one=6 neg_two=6 theta_option=optimal

echo "run setting CIFAR10 ripple MLPNet MSE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 1-2-3-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=1 pos_two=2 neg_one=3 neg_two=3 theta_option=optimal

echo "run setting CIFAR10 ripple MLPNet MSE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 ripple MLPProb BCE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinear MSE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 ripple MLPLinearProb BCE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 kernel MLPNet MSE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 kernel MLPProb BCE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinear MSE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting CIFAR10 kernel MLPLinearProb BCE 4-6-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=CIFAR10 method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=6 neg_one=5 neg_two=5 theta_option=optimal
