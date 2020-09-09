import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import math
    
class Model(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=12):
        super(Model, self).__init__()
        
        # The reason I do this is because initially loading the model would fail,
        # But if I do it again, somehow the model will load. This is probably due to
        # having a lower version of python.
        try:
            self.pretrained = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        except:
            try: 
                self.pretrained = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
            except:
                print("Loading model failed.")
                exit()
        self.num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(self.num_ftrs, 1000)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.pretrained(x)
        x = self.relu(x)
        x = self.l1(x)
        return x