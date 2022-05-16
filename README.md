# Visualizing vision models: From CNNs to ViT

[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/animeshbchowdhury/visualizing-vision-models) 
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview


## Installing dependencies

We recommend using [Anaconda](https://www.anaconda.com/) environment to install pre-requisites packages for running our framework and models.
We list down the packages which we used on our side for experimentations.

- cudatoolkit = 10.1
- numpy >= 1.20.1
- pandas >= 1.2.2
- pickleshare >= 0.7.5
- python >=3.9
- pytorch = 1.8.1
- scikit-learn = 0.24.1
- torch-geometric=1.7.0
- tqdm >= 4.56
- seaborn >= 0.11.1
- networkx >= 2.5
- joblib >= 1.1.0

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
	    ├── lambda_layer.py
	    ├── lambda_resnet.py
	    ├── metrics.py
	    ├── resnet.py
	    ├── train.py
	    └── utils.py


