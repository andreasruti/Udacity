# this file provides funcitons for train.py and predict.py

# Imports here
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict

from PIL import Image
import json

from workspace_utils import active_session




def get_loaders(data_dir):
    """
    Return dataloaders for training, validation and testing datasets.
    """
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'
    valid_dir = data_dir + '/valid/'

    data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(size = 224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                      std = [0.229, 0.224, 0.225])]),

    'valid': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(size = 224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                      std = [0.229, 0.224, 0.225])]),

    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])])}

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),

        'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),

        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64)

    class_to_idx = image_datasets['test'].class_to_idx

    return train_loader, valid_loader, test_loader, class_to_idx, image_datasets


def load_pretrained_model(arch):
    '''
    Load pretrained model
    '''

    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    else: #vgg19
        model = models.vgg19(pretrained = True)

    return model


def train_and_validate(device, train_loader, valid_loader, model, criterion, optimizer, epochs, print_every):
    '''
    Train the model and monitor the training process with validation loss and accuracy
    '''

    print('Start training...')

    steps = 0
    running_loss = 0
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(valid_loader):.3f}")

                running_loss = 0
                model.train()

    return(model)
    print('Training completed!')

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(size = 224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                      std = [0.229, 0.224, 0.225])]),

    'valid': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(size = 224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                      std = [0.229, 0.224, 0.225])]),

    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])])}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    # load image
    img = Image.open(image)

    # process image according to upper transformation and transform to numpy array
    img = np.array(data_transforms['test'](img))

    return img

def predict(device, image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file

    # process image
    image = process_image(image_path)

    # transform to tensor
    image = torch.from_numpy(image)

    # change size from [3, 224, 224] to [1, 3, 224, 224]
    image = image.unsqueeze(0)

    model.eval()

    # get top probs and classes and transform to numpy arrays
    with torch.no_grad():
        outputs = model(image)

        probs = torch.exp(outputs)

        top_p, top_class = probs.topk(topk, dim=1)

        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]

    # Transform into classes according to class_to_idx attached to checkpoint
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]

    return top_p, top_class

def transform_cat_to_name(json_file, classes):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[number] for number in classes]
    return class_names
