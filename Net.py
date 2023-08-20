import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch. autograd import Variable
classes=('Plane','Car  ','Bird ','Cat  ','Deer  ','Dog  ', 'Frog ', 'Horse', 'Ship ','Truck')
#NET STRCUTURE
class Net(nn.Module):
    def __init__(self, feat_1,feat_2,feat_3,feat_0=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(feat_0,feat_1,3)
        self.conv2 = nn.Conv2d(feat_1,feat_2,4)
        self.conv3 = nn.Conv2d(feat_2,feat_3,2)
        self.bn1 = nn.BatchNorm2d(feat_1)
        self.bn2 = nn.BatchNorm2d(feat_2)
        self.bn3 = nn.BatchNorm2d(feat_3)
        self.pool2= nn.AvgPool2d(2,2)     #AvgPool2d   MaxPool2d
        self.pool= nn.MaxPool2d(2,2) 
        self.dropout=nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(5*5*feat_3,500)
        self.fc2 = nn.Linear(500,300)
        self.fc3 = nn.Linear(300,100)
        self.fc4 = nn.Linear(100,10)
        self.ccc3= feat_3
    
        
    def forward(self,x): 
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))   #torch.relu   torch.sigmoid                                                          #self.dropout(x)
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))       #self.pool()
        x = self.bn3(F.relu(self.conv3(x)))             #self.bn3()
        x = x.view(-1, 5*5*self.ccc3)
        x = torch.relu(self.fc1(x))          # torch.tanh #10*F.softmax(self.hidden5(x)) nn.LogSoftmax(self.hidden5(x))
        #x = self.dropout(x)
        x=  torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  
       #x = nnn(self.fc3(x))         #when using nllloss
        x=self.fc4(x)
        return x