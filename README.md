# Visualizing vision models: From CNNs to ViT

[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/animeshbchowdhury/visualizing-vision-models) 
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)



## Overview

This work is performed as term-project for CSCI-GA 3033-091: Practical Deep Learning Systems at NYU Courant. In this project, we study and evaluate the impact of attention-based blocks on learned internal representations of vision models; varying from attention-free models (CNNs) to pure self-attention based models (ViT). We use state-of-the-art metrics: *Centered Kernel Alignment* (CKA) following the work of Raghu et al. (NeurIPS 2021) based on Hilbert-Schmidt Independence Criterion (HSIC) to understand the similarity across learned representations in various hidden layers. We performed our experiments on three state-of-the-art vision models namely: CNNs (*ResNet-18,34,50*), Hybrid networks (*Lambda ResNet-26,38,50*) and Vision Transformer (*ViT-large_patch32_224*). We evaluated the CNNs and hybrid networks on *CIFAR-100* dataset and ViT on *Imagenet* dataset.

Visualizing CNNs
![](https://github.com/animeshbchowdhury/visualizing-vision-models/blob/main/images/resnet.jpg)

Visualizing Lambda networks
![](https://github.com/animeshbchowdhury/visualizing-vision-models/blob/main/images/lambda.jpg)

Visualizing ViTs
![](https://github.com/animeshbchowdhury/visualizing-vision-models/blob/main/images/vit.jpg)
## Installing dependencies

We recommend using [Anaconda](https://www.anaconda.com/) environment to install pre-requisites packages for running our framework and models. We recommend installing the packages using *requirements.txt* file provided in our repository.

We list down the packages which we used on our side for experimentations.

- cudatoolkit = 11.3
- numpy >= 1.20.1
- pandas >= 1.2.2
- pickleshare >= 0.7.5
- python =3.9
- pytorch = 1.11.0
- scikit-learn = 0.24.1
- tqdm >= 4.56
- seaborn >= 0.11.1
- torchvision = 0.12
- timm

## Organisation

### Dataset directory structure

	├── images
	│   ├── lambda.jpg
	│   ├── resnet.jpg
	│   └── vit.jpg
	├── LICENSE
	├── notebooks
	│   ├── Lambda_R26_CIFAR100_PRETRAINED.ipynb
	│   ├── Lambda_R26_CIFAR100_RANDOM.ipynb
	│   ├── Lambda_R38_CIFAR100_PRETRAINED.ipynb
	│   ├── Lambda_R50_CIFAR100_PRETRAINED.ipynb
	│   ├── Lambda_R50_CIFAR100_RANDOM.ipynb
	│   ├── R101_CIFAR100.ipynb
	│   ├── R18_CIFAR100_PRETRAINED.ipynb
	│   ├── R18_CIFAR100_RANDOM.ipynb
	│   ├── R34_CIFAR100_PRETRAINED.ipynb
	│   ├── R34_CIFAR100_RANDOM.ipynb
	│   ├── R50_CIFAR100_PRETRAINED.ipynb
	│   ├── R50_CIFAR100_RANDOM.ipynb
	│   ├── ViT_ImageNet_PRETRAINED.ipynb
	│   └── ViT_ImageNet_RANDOM.ipynb
	├── README.md
	└── utilities
	    ├── lambda_layer.py                          # The file contains the definition of lambda layers for lambda resnets
	    ├── lambda_resnet.py			 # Utility file defining lambda network architecture
	    ├── metrics.py				 # Utility script computing Gaussian-CKA and RBF-CKA
	    ├── resnet.py				 # Utility script containing ResNet model architecture
	    ├── train.py                                 # Training script for CIFAR-100
	    └── utils.py                                 # Helper functions utility script for hooking,training and validation and CKA score computation

1. In ```notebooks``` directory, we have kept all the notebooks which were used to generate the visualization. The files having **random** as suffix denotes that the visualization for the model was generated with randomly initialized weights. Notebooks having **pretrained** suffix have visualizations with models trained on relevant dataset. One just needs to run the notebooks and get the results.

2. In ```utilities``` directory, we have kept various python utilities having model architecture design, metrics, training scripts and helper functions required for training and validation.



