import argparse
import os
import sys
import numpy as np
import logging
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
import random
import numpy as np
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
    aug_types = [torchvision.transforms.RandomRotation(40),                  
                torchvision.transforms.ColorJitter(brightness=([0.2, 1.8]), contrast=([0.5, 1.5]), saturation=([0.8, 1.2]), hue=([-0.5, 0.5])), 
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
    
    #print('Training data:', len(train_data))
    logger.info('Training data:', len(train_data))
    #print('Training Augmented data:', len(train_data_new))
    logger.info('Training Augmented data:', len(train_data_new))
    #print('Validation data:',len(val_data))
    logger.info('Validation data:',len(val_data))
    #print('Testing data:',len(test_data))
    logger.info('Testing data:',len(test_data))

    # The loaders with the augmented data
    train_loader = torch.utils.data.DataLoader(train_data_new, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

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
def get_accuracy(model, data_loader, criterion):
    
    correct = 0
    total = 0
    err = 0
    set_loss = 0
    
    use_cuda = True
    
    for imgs, labels in data_loader:
        
        
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################
        
        
        output = model(imgs)
        loss = criterion(output, labels)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]

        # return losses per epoch
        corr = output.max(1)[1] != labels
        err += int(corr.sum())
        set_loss += loss.item()

    return correct / total, float(err) / total, float(set_loss) / total

def evaluate(net, loader, criterion):

    use_cuda = True

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):

        inputs, labels = data

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        corr = outputs.max(1)[1] != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss


# Write the training code depending on the batch_size, learning rate and number of epochs
def train(model, args):
    
    torch.manual_seed(1000)

    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader = get_data_loader(args.data_dir, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    iters, losses, train_acc, val_acc, train_error, training_loss, valid_err, valid_loss = [], [], [], [], [], [], [], []
    
    use_cuda = True

    train_err = np.zeros(args.epochs)
    train_loss = np.zeros(args.epochs)
    val_err = np.zeros(args.epochs)
    val_loss = np.zeros(args.epochs)


#     # Check if checkpoints exists
#     if not os.path.isfile(args.checkpoint_path + '/checkpoint.pth'):
#         epoch_number = 0
#     else:    
#         model, optimizer, epoch_number = _load_checkpoint(model, optimizer, args) 

    # training
    n = 0 # the number of iterations
    start_time = time.time()
    for epoch in range(args.epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        i = 0
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
            n1, n2, n3 = get_accuracy(model, train_loader, criterion)         # compute training accuracy, err, loss
            train_acc.append(n1)
            train_error.append(n2)
            training_loss.append(n3)
            n4, n5, n6 = get_accuracy(model, val_loader, criterion)  # compute validation accuracy
            val_acc.append(n4)
            valid_err.append(n5)
            valid_loss.append(n6)
            n += 1
            i += 1

            # calculate stats per epoch for train and val loss
            corr = out.max(1)[1] != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion)
#         _save_checkpoint(model, optimizer, epoch, loss, args)
                    
        print(("Epoch {}: Train acc: {} | " + "Validation acc: {}").format(epoch, train_acc[epoch], val_acc[epoch]))
        logger.info(("Epoch {}: Train acc: {} | " + "Validation acc: {}").format(epoch, train_acc[epoch], val_acc[epoch]))

        #Manyally passed in string, need to change for other models
        model_path = get_model_name("resnet18", args.batch_size,args.learning_rate, epoch, args.sm_model_dir)
        torch.save(model.state_dict(), model_path)
    
#     _save_model(model, args.model_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Training Finished! Plotting Graphs...")
    logger.info("Total time elapsed: {:.2f} seconds".format(elapsed_time))

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
    plt.clf()

    plt.title("Training Error vs Validation Error Per Iteration")
    plt.plot(iters, train_error, label="Train")
    plt.plot(iters, valid_err, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'train_valid_err_iter.png'))
    plt.clf()

    plt.title("Training Loss vs Validation Loss Per Iteration")
    plt.plot(iters, training_loss, label="Train")
    plt.plot(iters, valid_loss, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.legend(loc='best')
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'train_valid_loss_iter.png'))
    plt.clf()

    plt.title("Train vs Validation Error") # per epoch
    n = len(train_err)
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'train_val_err.png'))
    plt.clf()

    plt.title("Train vs Validation Loss")   # per epoch
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    plt.savefig(os.path.join(args.output_data_dir,'train_val_loss.png'))

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    logger.info("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    logger.info("Final Validation Accuracy: {}".format(val_acc[-1]))

    logger.info("Graphs plotted...train() exited...")


# def _save_checkpoint(model, optimizer, epoch, loss, args):
#     print("epoch: {} - loss: {}".format(epoch, loss))
#     checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
#     print("Saving the Checkpoint: {}".format(checkpointing_path))
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         }, checkpointing_path)

    
# def _load_checkpoint(model, optimizer, args):
#     print("--------------------------------------------")
#     print("Checkpoint file found!")
#     print("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.pth'))
#     checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch_number = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
#     print('Resuming training from epoch: {}'.format(epoch_number+1))
#     print("--------------------------------------------")
#     return model, optimizer, epoch_number

# def _save_model(model, model_dir):
#     print("Saving the model.")
#     path = os.path.join(model_dir, 'model.pth')
#     # recommended way from http://pytorch.org/docs/master/notes/serialization.html
#     torch.save(model.cpu().state_dict(), path)


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
    
    ######################################## Code to train a model ########################################
    

    #Our Model
    resnet18=torchvision.models.resnet18(pretrained=True)
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
        
    train(resnet18, args)