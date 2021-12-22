from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os
from torchvision.io import read_image
from PIL import Image
import yaml


# Get configs from config file
stream = open("config.yaml", 'r')
config_dict = yaml.safe_load(stream)
batch_size = config_dict['batch_size']
learning_rate = config_dict['lr']
model_pth = config_dict['model_pth']
train_data = config_dict['train_data']
valid_data = config_dict['valid_data']
test_data = config_dict['test_data']

# reads json file
def show_cat_name(cat_num, print_bool=True):

    """

    :param cat_num: Category number for showing its name
    :param print_bool: Prints category name if True
    :return: If prints category is True, returns None else returns name of category
    """
    assert cat_num <= 102, "Category number must be in between 1 and 102"

    root_dir = '.'
    label_map_path = os.path.join(root_dir, 'cat_to_name.json')
    with open(label_map_path, 'r') as f:
        cat_to_name = json.load(f)
    cat_name = cat_to_name[str(cat_num)]
    if print_bool:
        print(f"Category {cat_num} is of {cat_name}")
    else:
        return cat_name


# gives training example count of cat_num
def no_examples(cat_num, print_bool=True, dataset='train'):
    assert dataset in ['train', 'valid'], "dataset = 'train' or 'valid'"

    root_dir = '.'
    cat_path = os.path.join(root_dir, dataset, str(cat_num))
    cat_name = show_cat_name(cat_num, print_bool=False)
    num_files = len([name for name in os.listdir(cat_path)])
    if print_bool:
        print(f"There are {num_files} images in category of {cat_name} ")
    else:
        return num_files


cat_examp = [{cat_num,no_examples(cat_num = cat_num,print_bool=False)} for cat_num in range(1,103)]
no_examp = [no_examples(cat_num = cat_num,print_bool=False) for cat_num in range(1,103)]


# shows image for cat_num
def show_im(cat_num, im_no=1, dataset='train', print_dim=True):
    assert dataset in ['train', 'valid'], "dataset = 'train' or 'valid'"

    no_examp = no_examples(cat_num, print_bool=False, dataset=dataset)
    assert im_no <= no_examp, "Image number out of range"

    root_dir = '.'
    cat_path = os.path.join(root_dir, dataset, str(cat_num))
    cat_name = show_cat_name(cat_num, print_bool=False)
    num_files = [name for name in os.listdir(cat_path)]
    im_path = os.path.join(cat_path, num_files[im_no])
    name_img = show_cat_name(cat_num=cat_num, print_bool=False) + " Image: " + str(im_no)
    image = read_image(im_path)
    image = image.numpy().transpose((1, 2, 0))
    if print_dim:
        print(f"The dimensions of image are {image.shape}")
    plt.imshow(image)
    plt.title(name_img)
    plt.axis('off')
    plt.show()


# Class for loading test dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_file, transform=None):
        self.path = path
        self.csv_file = csv_file

        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = self.csv_file.path[idx]
        img_name = self.csv_file.file_name[idx]
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0, img_name


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# saves model and returns its path.
def save_model(model, name="unknown", path=model_pth):
    """

    :param model: Model object to be saved
    :param name: Name of model
    :param path: Path where model is saved
    :return: Path of saved Model object
    """
    now = datetime.now()
    date, _ = str(now).split()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    _, time = dt_string.split()
    hr, minutes, sec = time.split(":")
    time_right = hr + "." + minutes
    name_model = name + '_' + time_right + '_' + date + '.pth'

    torch.save(model, os.path.join(path, name_model))
    return os.path.join(path, name_model)
