# Pytorch functions
import torch
# Neural network layers
import torch.nn as nn
# Optimizer
import torch.optim as optim
# Handling dataset
import torch.utils.data as data
# Torchvision library
import torchvision
import cv2
import copy

import argparse

from utils import *
from mlp_mixer_inspired import MLPMixer_Inspired
from mlp_mixer import MLPMixer

# Model trained on CIFAR-10 from scratch

def get_parser():
    parser = argparse.ArgumentParser(
        description='Mixer-MLP and Mixer-MLP-Inspired architectures for CIFAR10 images classification')
    # Dataset
    parser.add_argument('--inspired', action='store_true', help='Use this argument to load Mixer-MLP-Inspired model')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--patch_size', default=4, type=int) # Inspired uses 3-D patchs
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--n_epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    return parser

# Load parameters
parser = get_parser()
args = parser.parse_args()
print(args)


# Load CIFAR10 dataset
# Available datasets at https://pytorch.org/vision/stable/datasets.html
ROOT = './data'
train_data = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True)

# Data normalization
# Mean and std should be divided by 255 (maximum pixel value)
# because, after the ToTensor() transformation (see next step), images are normalized
# between 0 and 1.
train_mean = train_data.data.mean(axis=(0,1,2)) / 255
train_std = train_data.data.std(axis=(0,1,2)) / 255

# Compositions of transformations
train_transforms = torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(), # values are normalized between 0 and 1
                                                   torchvision.transforms.Normalize(train_mean, train_std)
])

test_transforms = torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(train_mean, train_std)
])

# Load data with transformations
train_data = torchvision.datasets.CIFAR10(root=ROOT,
                                        train=True,
                                        download=False,
                                        transform=train_transforms)

test_data = torchvision.datasets.CIFAR10(root=ROOT,
                                        train=False,
                                        download=False,
                                        transform=test_transforms)

# Split train data into train and validation sets
# (20% of training set will be used as validation set)
num_train_examples = int(len(train_data) * 0.8)
num_valid_examples = len(train_data) - num_train_examples

# Create 'Subset' objects
train_data, valid_data = data.random_split(train_data, [num_train_examples, num_valid_examples])

# Apply test transformations to the validation set
valid_data = copy.deepcopy(valid_data) # If we change train transformations, this won't affect the validation set
valid_data.dataset.transform = test_transforms

# Create iterators
BATCH_SIZE = args.batch_size

train_iterator = torch.utils.data.DataLoader(train_data, 
                                             shuffle=True, 
                                             batch_size=BATCH_SIZE)

valid_iterator = torch.utils.data.DataLoader(valid_data, 
                                             batch_size=BATCH_SIZE)

test_iterator = torch.utils.data.DataLoader(test_data, 
                                            batch_size=BATCH_SIZE)



if args.inspired:
    model_name = 'mlp-mixer-inspired-cifar10.pt'
    model = MLPMixer_Inspired(
        patch_size = args.patch_size, 
        output_dim = 10, 
        c = 3, 
        h = 32, 
        w = 32, 
        depth = args.depth
    )
else:
    model_name = 'mlp-mixer-cifar10.pt'
    model = MLPMixer(
        image_size = 32,
        patch_size = args.patch_size,
        dim = 512,
        depth = args.depth,
        num_classes = 10
    )    

# Loss
criterion = nn.CrossEntropyLoss() # Softmax + CrossEntropy

# Put model&criterion on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = criterion.to(device)
model = model.to(device)

# Optim
optimizer = create_optim(model.parameters(), args)

N_EPOCHS = args.n_epochs
train_losses, train_accs, valid_losses, valid_accs = model_training(N_EPOCHS, 
                                                                    model, 
                                                                    train_iterator, 
                                                                    valid_iterator, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device,
                                                                    model_name)

model_testing(model, test_iterator, criterion, device, model_name)

plot_results(N_EPOCHS, train_losses, train_accs, valid_losses, valid_accs, model_name)

