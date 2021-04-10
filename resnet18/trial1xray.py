import argparse
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from math import floor as fl
from PIL import Image
import pandas as pd
import random

# Load all datasets; we shall use 70% training, 15% validation and 15% testing samples
# Some helper functions to get the loaders for the training set, validation set and test set

def get_data_loader(dataset_main_path, batch_size):
    
    # Define a transform function that resizes images to 224x224
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                                  torchvision.transforms.ToTensor()])

    # Load training, validation and test data 
    train_data = torchvision.datasets.ImageFolder(dataset_main_path+'/Train', transform=transform)
    val_data = torchvision.datasets.ImageFolder(dataset_main_path+'/Validation', transform=transform)
    test_data = torchvision.datasets.ImageFolder(dataset_main_path+'/Test', transform=transform)

    # Training Data is augmented using three techniques
    aug_types = [torchvision.transforms.RandomRotation(random.randint(-10,10)),                  
                torchvision.transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)), 
                torchvision.transforms.RandomHorizontalFlip(1)]

    # Create augmented training data
    end_index_1 = 512
    end_index_2 = 1024
    end_index_3 = 1536

    train_indices = [list(range(0, end_index_1)), 
                    list(range(end_index_1, end_index_2)), 
                    list(range(end_index_2, end_index_3))]

    
    transform = torchvision.transforms.Compose([aug_types[0],
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor()])
    aug_dataset_1 = torchvision.datasets.ImageFolder(dataset_main_path+'/Train', transform=transform)
    aug_dataset_subset_1 = torch.utils.data.Subset(aug_dataset_1, train_indices[0])
    #train_data_new_1 = torch.utils.data.ConcatDataset([train_data, aug_dataset_subset_1])

    transform = torchvision.transforms.Compose([aug_types[1],
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor()])
    aug_dataset_2 = torchvision.datasets.ImageFolder(dataset_main_path+'/Train', transform=transform)
    aug_dataset_subset_2 = torch.utils.data.Subset(aug_dataset_2, train_indices[1])
    #train_data_new_2 = torch.utils.data.ConcatDataset([train_data, aug_dataset_subset])

    transform = torchvision.transforms.Compose([aug_types[2],
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor()])
    aug_dataset_3 = torchvision.datasets.ImageFolder(dataset_main_path+'/Train', transform=transform)
    aug_dataset_subset_3 = torch.utils.data.Subset(aug_dataset_3, train_indices[2])
    train_data_new = torch.utils.data.ConcatDataset([train_data, 
                                                     aug_dataset_subset_1, 
                                                     aug_dataset_subset_2, 
                                                     aug_dataset_subset_3])
    
    print('Training data:', len(train_data))
    print('Training Augmented data:', len(train_data_new))
    print('Validation data:',len(val_data))
    print('Testing data:',len(test_data))

    # The loaders with the augmented data
    train_loader = torch.utils.data.DataLoader(train_data_new, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch, model_dir):
    
    _path = os.path.join(model_dir, "model_{}_bs{}_lr{}_epoch{}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch))
    return _path

# Code to get the accuracy of both the training and validation set
def get_accuracy(model, data_loader):
    
    correct = 0
    total = 0
    
    use_cuda = True
    
    for imgs, labels in data_loader:
        
        
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        
        
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


# Write the training code depending on the batch_size, learning rate and number of epochs
def train(model, args):
    
    torch.manual_seed(1000)

    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader = get_data_loader(args.data_dir, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    iters, losses, train_acc, val_acc = [], [], [], []
    
    use_cuda = True

    # training
    n = 0 # the number of iterations
    for epoch in range(args.epochs):
        for imgs, labels in iter(train_loader):
            
           #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################
            
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/args.batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, train_loader)) # compute training accuracy 
            val_acc.append(get_accuracy(model, val_loader))  # compute validation accuracy
            n += 1
                    
        print(("Epoch {}: Train acc: {} | " + "Validation acc: {}").format(epoch, train_acc[epoch], val_acc[epoch]))

        #Manyally passed in string, need to change for other models
        model_path = get_model_name("resnet18", args.batch_size,args.learning_rate, epoch, args.sm_model_dir)
        torch.save(model.state_dict(), model_path)

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'loss.png'))
    plt.clf()

    plt.title("Training Curve")
    plt.figure(figsize=(10,10))
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations\nFinal Training Accuracy: {:0.2f}% | Final Validation Accuracy: {:0.2f}%".format(100*train_acc[-1], 100*val_acc[-1]))
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'accuracy.png'))
    
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


################ MAIN GUARD #####################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoint directories
    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    
    # Containe Environment Variables
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    # parse the arguments
    args = parser.parse_args()
    
#     # hyperparameters
#     epochs     = args.epochs
#     lr         = args.learning_rate
#     batch_size = args.batch_size
    
#     # write plots out
#     output_dir = args.images
#     # write model checkpoints out
#     model_dir  = args.model_dir
#     # read the data folders in (train, test, valid)
#     data_dir = args.data_dir
    
    ######################################## Code to train a model ########################################
    

    #Our Model
    resnet18 =torchvision.models.resnet18(pretrained=True)
    n_inputs = resnet18.fc.in_features

    resnet18.fc = nn.Sequential(
                        nn.Linear(n_inputs , 256),
                        nn.BatchNorm1d(256),
                        nn.Dropout(0.2),
                        nn.Linear(256 , 128),
                        nn.Linear(128 , 4))


    use_cuda = True  


    if use_cuda and torch.cuda.is_available():
        resnet18 = resnet18.to('cuda:0')
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')

    #proper model
#     train(data_dir, model_dir, output_dir, large_net, batch_size, epochs, lr)
    train(resnet18, args)