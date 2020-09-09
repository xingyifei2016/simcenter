import numpy as np
import pandas as pd
from scipy import misc
import glob
from collections import Counter
from pdb import set_trace as st
import tqdm
import matplotlib.pyplot as plt
import torch

def center_crop(inputs, size):
    assert inputs.shape[0] > size and inputs.shape[1] > size
    return inputs[int(inputs.shape[0] / 2) - int(size / 2):int(inputs.shape[0] / 2) + int(size / 2), int(inputs.shape[1] / 2) - int(size / 2):int(inputs.shape[1] / 2) + int(size / 2), :]

def get_dataloader(data_dir="/home/saschaho/roof_data/", percent_train=0.5, center_crop_size=224, train_batch=400, test_batch=400):
    df = pd.read_csv(data_dir+'roofMaterial_Simplified.csv')
    unique_classes = pd.unique(df.RoofMateria)
    class_dict = {}
    for idx, j in enumerate(unique_classes):
        class_dict[j] = idx
    print("Label distribution (Before Train/Test Split) :")
    
    # This counts the instance of each class in the dataset
    class_counter = Counter(df.RoofMateria.values)
    print(class_counter)
    
    # Convert class_counter keys to be purely numeric
    class_counter_numeric = {}
    for i in class_counter:
        class_counter_numeric[class_dict[i]] = int(class_counter[i])
        
    # This counts the number of instances of each class to use for training.
    # For additional instances of that class, we would use it as testing.
    train_counter = {}
    for i in class_counter:
        train_counter[class_dict[i]] = int(class_counter[i] * percent_train)
    
    # This counts the current number of instances of each class used for training.
    current_counter = {}
    for i in class_counter:
        current_counter[class_dict[i]] = 0
        
    print("##########")
    print("Loading labels...")
    
    labels = []
    for roof_material in df.RoofMateria.values:
        labels.append(class_dict[roof_material])
    print("Finished processing labels")
    print("##########")
    print("Loading images...")
    
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []
    
    for idx, image_path in enumerate(tqdm.tqdm(glob.glob(data_dir+'images/*.png'))):
        image = misc.imread(image_path)
        image = center_crop(image, center_crop_size)
        index = int(image_path.split("/")[-1].split(".")[0])-1
        class_type = labels[index]
        
        if current_counter[class_type] < train_counter[class_type]:
            current_counter[class_type] += 1
            train_X.append(image)
            train_Y.append(class_type)
        else:
            test_X.append(image)
            test_Y.append(class_type)
    print("Finished processing images")
    
    
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    
    # Normalize for Resnet, check https://pytorch.org/hub/pytorch_vision_resnet/

    train_X = ( train_X - np.array([[0.485, 0.456, 0.406]]) ) / np.array([[0.229, 0.224, 0.225]])
    test_X = ( test_X - np.array([[0.485, 0.456, 0.406]]) ) / np.array([[0.229, 0.224, 0.225]])
    
    train_X = np.transpose(train_X, (0, 3, 1, 2))
    test_X = np.transpose(test_X, (0, 3, 1, 2))
    # Generate weights for weighted sampler
    train_weights = []
    for labels in train_Y:
        assert(1./train_counter[labels] > 0 and 1./train_counter[labels] < 1)
        train_weights.append(1./train_counter[labels])
    train_weights = np.array(train_weights)
    
    test_weights = []
    for labels in test_Y:
        assert(1./(class_counter_numeric[labels] - train_counter[labels]) > 0 and 1./(class_counter_numeric[labels] - train_counter[labels]) < 1)
        test_weights.append(1./(class_counter_numeric[labels] - train_counter[labels]))
    test_weights = np.array(test_weights)
    
    # Weighted sampler creation
    train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights))
    test_sampler = torch.utils.data.WeightedRandomSampler(test_weights, len(test_weights))
    
    # Create torch dataset
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X).type(torch.FloatTensor), torch.from_numpy(train_Y).type(torch.LongTensor))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(test_X).type(torch.FloatTensor), torch.from_numpy(test_Y).type(torch.LongTensor))
    
    # Create final dataloader
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=train_batch, num_workers=4, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=test_batch, num_workers=4, sampler=test_sampler)
    
    return train_loader, test_loader
    
    
    
    