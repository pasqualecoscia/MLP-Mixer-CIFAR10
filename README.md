# MLP-Mixer-CIFAR10
MLP-Mixer architectures for classifying CIFAR10 images

This repo contains the implementation of the MLP-Mixer architecture
and another MLP-Mixer-inspired model to classify CIFAR-10 images
pretraining the model on the TinyImageNet dataset.

Steps:
1. Download the TinyImageNet dataset:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

2. Run main.py

```
python3 main.py
```

Results:

MLP-Mixer Accuracy: 

MLP-Mixer-Inspired Accuracy: 

Optional parameters can be provided as input (e.g., batch_size, dep-th, n_epochs, ...).
Open the main.py file for the complete list of input arguments.
