import numpy as np
import time
import pandas as pd
import os, random
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision
import torchvision.models as models
torch.utils.model_zoo
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
import json
from PIL import Image
from torch.autograd import Variable

#Using argparse variable
parser = argparse.ArgumentParser()
parser.add_argument('-ch','--checkpoint',action='store',help='Filepath of the previously made checkpoint',default='checkpoint.pth')
parser.add_argument('-g','--gpu',action='store_true',help='GPU used for prediction',default=True)
parser.add_argument('-f','--filepath',dest='filepath',help='Filepath of the image to be classified',default='/home/workspace/ImageClassifier/flowers/test/5/image_05169.jpg')
parser.add_argument('-t','--top_k',metavar='',help='Number of flower species with top probabilities',default=5,type=int)
parser.add_argument('-c','--cname_json',metavar='',help='Filepath of the json file to be used for category mapping of the flower species', default='cat_to_name.json')
args=parser.parse_args()

with open(args.cname_json,'r') as f:
    cat_to_name=json.load(f)
    
#Loading checkpoint  
def load_checkpoint(file_path):
    checkpoint=torch.load(file_path)
    model=getattr(torchvision.models,checkpoint['Arch'])(pretrained=True)
    
    #Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier=nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(checkpoint['Input_Size'],checkpoint['Hidden_Layers'])),
                        ('relu1',nn.ReLU()),
                        ('drop',nn.Dropout(checkpoint['Drop'])),
                        ('fc2',nn.Linear(checkpoint['Hidden_Layers'],checkpoint['Output_Size'])),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.classifier.epochs = checkpoint['Epochs']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#Image Preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Get the dimensions of the image
    w, h = image.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    image = image.resize((255, int(255*(h/w))) if w < h else (int(255*(w/h)), 255))
    
    # Get the dimensions of the new image size
    w, h = image.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (w - 224)/2
    top = (h - 224)/2
    right = (w + 224)/2
    bottom = (h + 224)/2
    image = image.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    image = np.array(image)
    
    # Make all values between 0 and 1
    image = image/255
    
    # Normalize based on the preset mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image=(image-mean)/std
    
    # Make the color channel dimension first instead of last
    image = image.transpose((2, 0, 1))
    return image

#Class prediction
# Using our model to predict the label
def predict(path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
        
    im = Image.open(path)
    array = process_image(im)
    tensor = torch.from_numpy(array)
    
    #Converting to object of type torch.FloatTensor
    if cuda:
        conv = Variable(tensor.float().cuda())
    else:       
        conv = Variable(tensor)
    
    conv = conv.unsqueeze(0)
    # Pass the image through our model
    conv_f=model.forward(conv)
    # Reverse the log function in our output
    #topk(k) method returns both the highest k probabilities and the indices of those probabilities    corresponding to the classes
    ps=torch.exp(conv_f).data.topk(topk)
    #ps[0] consist of highest k probabilities
    prob=ps[0].cpu()
    #ps[1] consist of class name of the highest k probabilities
    classes=ps[1].cpu()   
    #Inverting the dictionary so as to get mapping from index to class as well.
    index_to_class={model.class_to_idx[key]: key for key in model.class_to_idx}
    m_idx_c = list()
    
    for ll in classes.numpy()[0]:
        m_idx_c.append(index_to_class[ll])     
    return prob.numpy()[0], m_idx_c

model=load_checkpoint(args.checkpoint)
gpu=args.gpu

image_path=args.filepath
prob,classes=predict(image_path,model,args.top_k)
print('\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* TOP K CLASSES *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print('Probabilities: ',prob)
print('Flower Species: ',[cat_to_name[x] for x in classes])
print('Species Index: ',classes)

#Saving the classification plot as predict.png
max_idx = np.argmax(prob)
max_prob = prob[max_idx]
label = classes[max_idx]
print('\033[1m' + '\nFlower Species With Highest Probability: ',cat_to_name[label])
print('\033[0m')
#Defining the probability plot
f_1 = plt.figure(figsize=(7,7))
axis_1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
axis_2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
axis_1.axis('off')
ll = []
for class_idx in classes:
    #Appending the labels of the class integer into empty list
    ll.append(cat_to_name[class_idx])
y_pos = np.arange(5)
plt.barh(y_pos, prob, xerr=0, align='center', color='green')
plt.title('Prediction Plot')
plt.yticks(y_pos,ll)
plt.xlabel('Probabilities -->')
plt.ylabel('Predicted flower species -->')
axis_2.invert_yaxis()
plt.savefig('/home/workspace/prediction.png')
print('Done!! Classification plot is saved in /home/workspace/prediction.png')