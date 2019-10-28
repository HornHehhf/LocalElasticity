#!/usr/bin/env bash
echo "run setting MNIST ripple CNN_MNIST MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 5-8-9-3 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=5 pos_two=8 neg_one=9 neg_two=3 theta_option=optimal

echo "run setting MNIST ripple CNN_MNIST MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 4-9-5-7 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=4 pos_two=9 neg_one=5 neg_two=7 theta_option=optimal

echo "run setting MNIST ripple CNN_MNIST MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 7-9-5-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=7 pos_two=9 neg_one=5 neg_two=4 theta_option=optimal

echo "run setting MNIST ripple CNN_MNIST MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 5-9-8-4 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=5 pos_two=9 neg_one=8 neg_two=4 theta_option=optimal

echo "run setting MNIST ripple CNN_MNIST MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 3-5-8-9 optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=3 pos_two=5 neg_one=8 neg_two=9 theta_option=optimal

echo "run setting MNIST ripple CNN_MNIST MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=CNN_MNIST loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST ripple ResNet MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=ripple \
model_option=ResNet loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel CNN_MNIST MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=CNN_MNIST loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
echo "run setting MNIST kernel ResNet MSE 3-8-5-5 optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_clustering.py dataset=MNIST method_option=kernel \
model_option=ResNet loss_option=MSE pos_one=3 pos_two=8 neg_one=5 neg_two=5 theta_option=optimal
