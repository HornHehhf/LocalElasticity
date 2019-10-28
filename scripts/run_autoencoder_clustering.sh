#!/usr/bin/env bash
echo "running autoencoder clustering 5 8 9 3"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=5 pos_two=8 neg_one=9 neg_two=3
echo "running autoencoder clustering 4 9 5 7"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=4 pos_two=9 neg_one=5 neg_two=7
echo "running autoencoder clustering 7 9 5 4"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=7 pos_two=9 neg_one=5 neg_two=4
echo "running autoencoder clustering 5 9 8 4"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=5 pos_two=9 neg_one=8 neg_two=4
echo "running autoencoder clustering 3 5 8 9"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=3 pos_two=5 neg_one=8 neg_two=9
echo "running autoencoder clustering 3 8 5 5"
CUDA_VISIBLE_DEVICES=1 python sources/autoencoders_clustering.py dataset=MNIST pos_one=3 pos_two=8 neg_one=5 neg_two=5
