#!/usr/bin/env bash
# Run local elasticity based clustering on MNIST

echo "run setting MNIST ripple MLPNet MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal

echo "run setting MNIST ripple MLPNet MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal

echo "run setting MNIST ripple MLPNet MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal

echo "run setting MNIST ripple MLPNet MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal

echo "run setting MNIST ripple MLPNet MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal

echo "run setting MNIST ripple MLPNet MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPNet loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST ripple MLPProb BCE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPProb loss_option=BCE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST ripple MLPLinear MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinear loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST ripple MLPLinearProb BCE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=MLPLinearProb loss_option=BCE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel MLPNet MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPNet loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel MLPProb BCE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPProb loss_option=BCE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel MLPLinear MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinear loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel MLPLinearProb BCE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=MLPLinearProb loss_option=BCE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
