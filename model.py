import yaml
import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets,models
import math
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from PIL import Image
from datetime import datetime

from utils import *
data_dir = '.'
test_path = os.path.join(data_dir, 'test')
sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
sample_sub['path'] = sample_sub['file_name'].apply(lambda x: os.path.join(test_path, x))

# Get configs from config file
stream = open("config.yaml", 'r')
config_dict = yaml.safe_load(stream)
batch_size = config_dict['batch_size']
learning_rate = config_dict['lr']
model_pth = config_dict['model_pth']
train_data = config_dict['train_data']
valid_data = config_dict['valid_data']
test_data = config_dict['test_data']

# Apply transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomRotation(30,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'valid': transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
}

# Load dataloaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle= True, num_workers=0)
              for x in ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Trains Model
def train_model2(model, criterion, optimizer,
                 num_epochs=3, dataloaders= dataloaders, print_progress=False):

    """

    :param model: Model type object
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param num_epochs: Number of epochs
    :param dataloaders: Dataloaders, must be a dictionary having train and val as keys
    :param print_progress: prints progress if true
    :return: trained model object
    """

    min_val_loss = np.Inf
    best_model_wts = copy.deepcopy(model.state_dict())

    since = time.time()
    best_epoch = -1

    for epoch in range(num_epochs):
        valid_loss = 0.0
        train_loss = 0.0
        model.train()
        running_corrects = 0

        for iter1, (inputs, labels) in enumerate(dataloaders['train']):

            inputs = inputs.to(device)
            inputs = inputs.type(torch.float)
            labels = labels.to(device)
            labels = labels.type(torch.long)

            optimizer.zero_grad()
            out = model(inputs)
            _, preds = torch.max(out, 1)
            # out = torch.mul(out,100)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            if print_progress:
                print(
                    f"Epoch: {epoch}\t{100 * (iter1 + 1) / len(dataloaders['train']):.2f}" + '%',
                    end='\r')


        else:
            print()
            with torch.no_grad():
                model.eval()

            for iter2, (inputs, labels) in enumerate(dataloaders['valid']):
                inputs = inputs.to(device)
                inputs = inputs.type(torch.float)
                labels = labels.to(device)
                labels = labels.type(torch.long)

                output1 = model(inputs)
                _, preds1 = torch.max(output1, 1)
                # output1 = torch.mul(output1,100).to(device)
                loss = criterion(output1, labels)
                valid_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds1 == labels.data)
                print(
                    f'Epoch: {epoch}\t{100 * (iter2 + 1) / len(dataloaders["valid"]):.2f} %',
                    end='\r')
        len_train1 = 6552
        len_val1 = len(dataloaders['valid'].dataset)
        train_loss = train_loss / len_train1
        valid_loss = valid_loss / len_val1
        if print_progress:
            print(
                f'\nEpoch: {epoch + 1} \tTraining Loss: {math.sqrt(train_loss):.4f} \tValidation Loss: {math.sqrt(valid_loss):.4f}')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'Accuracy : {100 * running_corrects / len_val1} %')
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        print('Best val Loss: {:4f}'.format(math.sqrt(min_val_loss)))
        print(f'Epoch completed: {epoch+1}')
        print(f'Best Epoch: {best_epoch+1}')

    model.load_state_dict(best_model_wts)
    return model


def process_image(img_path):

    """
        :param img_path: Path of image to be processed
        :returns processed numpy array
        Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a Numpy array
    """
    img = Image.open(img_path)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop Image

    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))

    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    return img

# Load test dataset from class defined in utils


test_dataset = TestDataset(data_dir+'test', sample_sub,data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load Class to idx dictionary
class_to_idx = image_datasets['valid'].class_to_idx
idx_to_class = {val: key for key, val in class_to_idx.items()}


def predict(model_path, dataloader, print_progress=False):

    """

    :param model_path: Path of Model used for prediction
    :param dataloader: Test DataLoader
    :param print_progress: Prints progress if True
    :return: Prediction(as a list) on test folder defined by config file
    """

    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for ii, (images, _, img_names) in enumerate(dataloader, start=1):

            if print_progress:
                if ii % 5 == 0:
                    print('Batch {}/{}'.format(ii, len(dataloader)))
            images = images.to(device)
            logps = model(images)
            ps = torch.exp(logps)

            # Top indices
            _, top_indices = ps.topk(1)
            top_indices = top_indices.detach().cpu().numpy().tolist()

            # Convert indices to classes
            top_classes = [idx_to_class[idx[0]] for idx in top_indices]
            # print("Img:" ,img_names)
            for i, img_name in enumerate(img_names):
                predictions[img_name] = top_classes[i]

        print('\nPrediction Generation Completed')

    return predictions
