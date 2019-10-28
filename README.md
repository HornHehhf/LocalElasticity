# Local Elasticity
This is the code repository for the ArXiv paper [The Local Elasticity of Neural Networks](https://arxiv.org/pdf/1910.06943.pdf).
If you use this code for your work, please cite
```
@article{he2019local,
  title={The Local Elasticity of Neural Networks},
  author={He, Hangfeng and Su, Weijie J},
  journal={arXiv preprint arXiv:1910.06943},
  year={2019}
}

```

## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
pytorch(https://pytorch.org)

## Change the Dir Path
Change the /path/to/experiments/dir to your experiments dir in autoencoders_clustering.py, elastic_effect_clustering.py, elastic_effect_synthetic.py and train_models.py\
You need to create corresponding directories in your experiments dir, such as models, data and figures.

## Reproducing experiments

To reproduce our simulations
```
sh scripts/run_elastic_effect_synthetic.sh
```

To reproduce the results of the local elasticity based clustering on MNIST
```
sh scripts/run_train_models_MNIST.sh
sh scripts/run_elastic_effect_clustering_MNIST.sh
```

To reproduce the results of the local elasticity based clustering on CIFAR10
```
sh scripts/run_train_models_CIFAR10.sh
sh scripts/run_elastic_effect_clustering_CIFAR10.sh
```

To reproduce the results of different architectures (CNN and ResNet) based clustering on MNIST
```
sh scripts/run_train_models_architectures.sh
sh scripts/run_elastic_effect_clustering_architectures.sh
```

To reproduce the results of autoencoder based clustering on MNIST
```
sh scripts/run_autoencoder_clustering.sh
```

To reproduce the results of pre-trained ResNet152 based clustering on MNIST
```
Uncomment clustering_images_resnet152(train_data, index_list, train_data_label[index_list]) in sources/elastic_effect_clustering.py
```
