#-------------------------------------------------------------------------------
# import packages
#-------------------------------------------------------------------------------
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F

from PIL import Image

import argparse
import utils

#-------------------------------------------------------------------------------
# input arguments
#-------------------------------------------------------------------------------
def get_input_args():
    '''
    user arguments
    '''

    parser = argparse.ArgumentParser(description='Predict flower')
    parser.add_argument('img', type=str, help='Path to image file to predict')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint to be loaded')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top classes to predict')
    parser.add_argument('--category_names', type=str, default='None', help='Path to JSON file for mapping class values to category names')
    #parser.add_argument('--gpu', type=bool, default=False, help='Set to True to run on GPU')
    parser.add_argument('--gpu', action='store_true', help='To run on GPU')

    return parser.parse_args()

args = get_input_args()
print(args)

# check
if(args.topk<=1 and args.topk>102):
    print("Error: Invalid top K value")
    print("Must be between 1 and 102")
    quit()


#-------------------------------------------------------------------------------
# Classifier class (for load_chekpoint function)
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

    # According problem: https://knowledge.udacity.com/questions/195825
    # Put load_checkpoint function into class Classifier
    def load_checkpoint(self, filepath):
        '''
        load checkpoint and rebuild model
        '''

        # According problem: https://knowledge.udacity.com/questions/237748
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        #print(checkpoint)

        # load pre-trained network VGG16
        model = models.vgg16(pretrained=True) if checkpoint['arch'] == 'vgg16' else models.vgg19(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']

        model.load_state_dict(checkpoint['state_dict'])

        return model


#-------------------------------------------------------------------------------
# load checkpoint
#-------------------------------------------------------------------------------
input_size = 1000
hidden_size = 100
output_size = 102
p_dropout = 0.5

x = Classifier()
model = x.load_checkpoint(args.checkpoint)

#-------------------------------------------------------------------------------
# predict top k classes
#-------------------------------------------------------------------------------

# run prediction
device = torch.device("cuda" if args.gpu else "cpu")
if device and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
probs, classes = utils.predict(device, args.img, model, args.top_k)

# change classes to flower names if needed
if args.category_names != 'None':
    classes = utils.transform_cat_to_name(args.category_names, classes)
prediction = dict(zip(classes, probs))

# show results
print(prediction)
