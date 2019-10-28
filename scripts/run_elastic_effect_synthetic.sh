#!/usr/bin/env bash

# Run simulations
echo "run setting: torus ripple MLPNet MSE optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_synthetic.py data_option=donut method_option=ripple \
model_option=MLPNet loss_option=MSE theta_option=optimal
echo "run setting: torus ripple MLPLinear MSE optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_synthetic.py data_option=donut method_option=ripple \
model_option=MLPLinear loss_option=MSE theta_option=optimal

echo "run setting: two folded boxes ripple MLPNet3Layer MSE optimal"
CUDA_VISIBLE_DEVICES=1 python sources/elastic_effect_synthetic.py data_option=two_fold_surface method_option=ripple \
model_option=MLPNet3Layer loss_option=MSE theta_option=optimal
echo "run setting: two folded boxes ripple MLPLinear3Layer MSE optimal"
CUDA_VISIBLE_DEVICES=0 python sources/elastic_effect_synthetic.py data_option=two_fold_surface method_option=ripple \
model_option=MLPLinear3Layer loss_option=MSE theta_option=optimal

