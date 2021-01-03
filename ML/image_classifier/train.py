#-------------------------------------------------------------------------------
# import packages
#-------------------------------------------------------------------------------
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F

from workspace_utils import active_session

import argparse
import utils

#-------------------------------------------------------------------------------
# input arguments
#-------------------------------------------------------------------------------
def get_input_args():
    '''
    user arguments
    '''

    parser = argparse.ArgumentParser(description='Train model to classify flowers')
    parser.add_argument('data_dir', type=str, help='Directory where image data is stored')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory where checkpoints will be saved')
    parser.add_argument('--arch', type=str, default='vgg16', help='Choose architecture (vgg16 or vgg19)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Choose learning rate for model training')
    parser.add_argument('--hidden_units', type=int, default=500, help='Choose number of hidden units in the model')
    parser.add_argument('--epochs', type=int, default=3, help='Choose number of epochs for training the model')
    parser.add_argument('--gpu', action='store_true', help='To run on GPU')

    # Store user options in the "args" variable
    return parser.parse_args()

# get user arguments
args = get_input_args()
print(args)

# checks
if(not(args.learning_rate>0 and args.learning_rate<1)):
    print("Error: Invalid learning rate")
    print("Must be between 0 and 1 exclusive")
    quit()

if(args.epochs<=0):
    print("Error: Invalid epoch value")
    print("Must be greater than 0")
    quit()

if(args.hidden_units<=0):
    print("Error: Invalid number of hidden units given")
    print("Must be greater than 0")

if args.arch not in ["vgg16", "vgg19"]:
    print("Error: invalid architecture name received")
    print("Type \"python train.py --help\" for more information")
    quit()
    

#-------------------------------------------------------------------------------
# Classifier class
#-------------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5000)
        self.fc2 = nn.Linear(5000, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # dropout module with 0.5 drop prob
        self.dropout = nn.Dropout(p = p_dropout)

    def forward(self, x):
        # flatten
        x = x.view(x.shape[0], -1)

        # input and hidden layers (with dropout)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # output (without dropout)
        x = F.log_softmax(self.fc3(x), dim = 1)

        return x


#-------------------------------------------------------------------------------
# get model
#-------------------------------------------------------------------------------

# load data
train_loader, valid_loader, test_loader, class_to_idx, image_datasets = utils.get_loaders(data_dir=args.data_dir)

# load pretrained network
model = utils.load_pretrained_model(arch=args.arch)

#-------------------------------------------------------------------------------
# set hyperparams for model training
#-------------------------------------------------------------------------------
for param in model.parameters():
    param.requires_grad = False
input_size = model.classifier[0].in_features
hidden_size = args.hidden_units
output_size = 102
p_dropout = 0.5
model.classifier = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)


#-------------------------------------------------------------------------------
# actual training
#-------------------------------------------------------------------------------
device = torch.device("cuda" if args.gpu else "cpu")
if device and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
model.to(device);
with active_session():
    utils.train_and_validate(device, train_loader, valid_loader, model, criterion, optimizer, epochs = args.epochs, print_every = 20)


#-------------------------------------------------------------------------------
# save checkpoint
#-------------------------------------------------------------------------------
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'arch': args.arch,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, args.save_dir+"/checkpoint.pth")
print('Checkpoint saved!')
