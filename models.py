## TODO: define the convolutional neural network architecture
num_output = 136 # As it's suggest final linear layer have to output 136 values, 2 for each of the 68 keypoint (x,y) pairs.
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## Define layers of a CNN
        ## 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        # Output of convulation layer would have height and width of 3 and depth of 128
        self.fc1 = nn.Linear(28*28*128, num_output)
        #self.fc2 = nn.Linear(10000, num_output)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x):
        #print("Enters Forward")
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        
        # flatten the input image
        #x = x.view(x.size(0), -1) same as x.view(-1, 28x28x128)
        x = x.view(-1, 28*28*128)
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #x = self.fc2(x)
        #print(x.shape)

        #print("Forwarded")
        return x
