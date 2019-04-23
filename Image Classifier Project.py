#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[1]:


# Imports here
# Package Imports
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import numpy as np
import time
import pandas as pd
import os, random
import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torchvision
import torchvision.models as models
torch.utils.model_zoo
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
import json
from PIL import Image
from torch.autograd import Variable


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
#Training_data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

#Validation_data
data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

#Testing_data
data_transforms_testing = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])




# TODO: Load the datasets with ImageFolder
#Training_data
image_datasets_train =datasets.ImageFolder(train_dir,transform= train_transforms) 

#Validation_data
image_datasets_validation =datasets.ImageFolder(valid_dir,transform=data_transforms_validation)

#Testing_data
image_datasets_testing =datasets.ImageFolder(test_dir,transform=data_transforms_testing)




# TODO: Using the image datasets and the trainforms, define the dataloaders
#Training_data
dataloaders_train =torch.utils.data.DataLoader(image_datasets_train,batch_size=32,shuffle=True)

#Validation_data
dataloaders_validation =torch.utils.data.DataLoader(image_datasets_validation,batch_size=32)

#Testing_data
dataloaders_testing =torch.utils.data.DataLoader(image_datasets_testing,batch_size=32)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[5]:


# TODO: Build and train your network
#Pre-Trained network
model = models.vgg16(pretrained=True)

#Freeze parameters
for param in model.parameters():
    param.requires_grad = False

#Hyperparameters
input_size = 25088
output_size = 102
hidden_layers = [1500, 102]
drop = 0.2

#Defining the architecture of classifier Neural network 
classifier=nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(input_size,hidden_layers[0])),
                        ('relu1',nn.ReLU()),
                        ('drop',nn.Dropout(drop)),
                        ('fc2',nn.Linear(hidden_layers[0],hidden_layers[1])),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

#Replacing the vgg16's inbuilt classifier with our own coded custom classifier 
model.classifier = classifier

steps=0
print_every=4
epochs = 2

criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(), lr=0.0001)


#Shifting the training process to Cuda GPU so as to process faster
model.to('cuda')

for i in range(epochs):
    running_loss=0
    for images,labels in dataloaders_train:
        steps +=1
        images, labels = images.to('cuda'), labels.to('cuda')#Shifting the images and labels to Cuda GPU
        optimizer.zero_grad()
        output=model.forward(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        
        if steps % print_every == 0:
            print("Epoch= {}/{}    ".format(i+1, epochs),
                  "Training Loss= {:.4f}".format(running_loss/print_every))    
        running_loss = 0
        
        #Calculating the accuracy
        if steps % 20*print_every == 0:
            corr = 0
            tot = 0
            for ii, (images, labels) in enumerate(dataloaders_validation):
                with torch.no_grad():
                    images, labels = images.to('cuda'), labels.to('cuda')
                    
                # Forward and backward passes
                    output = model.forward(images)
                    _, predicted = torch.max(output.data, 1)
                    corr += (predicted == labels).sum().item()
                    tot += labels.size(0)
                    
            print("Accuracy: {0:.2f}%".format(corr/tot*100))
            


# # Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[6]:


accuracy=0
t=0
corr=0
print("Testing trained network on test data")
for ii, (images, labels) in enumerate(dataloaders_testing):
    with torch.no_grad():
        images,labels=images.to('cuda'),labels.to('cuda')
        output=model.forward(images)
        _,pred=torch.max(output.data,1)
        corr+=(pred==labels).sum().item()
        ps=torch.exp(output)
        equality=(labels.data==ps.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
        t+=labels.size(0)
print("Total Images=",t)
print("Correctly Classified Images=",corr)
print("Accuracy={0:.2f}%".format(corr/t*100))        


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[7]:


# TODO: Save the checkpoint
model.class_to_idx = image_datasets_train.class_to_idx

checkpoint={'Input_Size':input_size,
              'Output_Size':output_size,
              'Hidden_Layers':1500,
              'Drop':drop,
              'Epochs':epochs,
              'Arch':'vgg16',
              'Loss':loss,
              'model_state_dict':model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'class_to_idx':model.class_to_idx}
torch.save(checkpoint,'checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[8]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
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


# In[9]:


model=load_checkpoint('checkpoint.pth')


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[10]:


# Process our image
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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[11]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)    
    return ax


# In[12]:


#Show Processed Picture
img=random.choice(os.listdir('./flowers/test/5/'))
img_path ='./flowers/test/5/' + img

with Image.open(img_path) as image:
    imshow(process_image(image))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[13]:


# Using our model to predict the label
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
        
    im = Image.open(image_path)
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
    #topk(k) method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes
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


# In[14]:


image=random.choice(os.listdir('./flowers/test/5/'))
image_path='./flowers/test/5/' + image
with  Image.open(image_path) as img:
    plt.imshow(img)
    
prob, classes = predict(image_path, model)
print(prob)
print([cat_to_name[x] for x in classes])
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[15]:


# TODO: Display an image along with the top 5 classes
prob, classes = predict(image_path, model)
max_idx = np.argmax(prob)
max_prob = prob[max_idx]
label = classes[max_idx]
    
#Defining the probability plot
f_1 = plt.figure(figsize=(7,7))
axis_1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
axis_2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
#Opening the image
img = Image.open(image_path)
axis_1.axis('off')
#Mentioning the flower name of the image
axis_1.set_title(cat_to_name[label]) 
#Displaying the plot
axis_1.imshow(img)
ll = []
for class_idx in classes:
    #Appending the labels of the class integer into empty list
    ll.append(cat_to_name[class_idx])
y_pos = np.arange(5)
axis_2.set_yticks(y_pos)
axis_2.set_yticklabels(ll)
axis_2.set_xlabel('Probabilities -->')
axis_2.set_ylabel('Predicted flower species -->')
axis_2.invert_yaxis()
axis_2.barh(y_pos, prob, xerr=0, align='center', color='green')
plt.show()


# 
