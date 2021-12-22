# Flower-Classifier
Classifier models trained on 102 Category flower dataset by Oxford

## DataSet
102 Category flower dataset released by Oxford. 
The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. 
In addition, there are categories that have large variations within the category and several very similar categories.

Original Dataset Website - > http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

I thank Professor Plum, Ph.D. for providing this dataset.

## Requirements
* Python >= 3.6
* PyTorch >= 1.0.0
* torchvision
* Pandas >=1.2.0

## Training
The following models were trained:
* AlexNet
* DenseNet201
* GoogLeNet
* MobileNet V2
* ResNet34

A linear layer was added to every model pretrained on ImageNet with dimension (no_in_features, no_of_classes).
Here, no_in_features = 1000 and no_of_classes = 102.

## Usage:
To get the trained models use the kaggle dataset - https://www.kaggle.com/apoorvbhardwaj/102-category-oxford-models
For making prediction with the model, call 'predict' function defined in model.py 

To get the training, validation and test data:
refer to kaggle competition - https://www.kaggle.com/c/oxford-102-flower-pytorch/ or the original dataset website - http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

For downloading the dataset use the Kaggle API.   
Documentation of Kaggle API - https://github.com/Kaggle/kaggle-api

Change the path of dataset in config file.

For training a model, define the model in main.py and run the script. 
Write the model path in config file, where you want the model to be saved.
Learning rate and batch size can be changed using config file.

## Results

| Model      | Accuracy | Public Score |
|------------|----------|--------------|
|AlexNet     | 64.87%   | 69.19%       |
|DenseNet201 | 82.19%   | 84.35%       |
|GoogLeNet   | 76.34%   | 83.12%       |  
|MobileNet V2| 79.26%   | 79.95%       |
|ResNet34    | 76.58%   | 80.44%       |

Accuracy was obtained by submitting the submissions file to the kaggle competition - https://www.kaggle.com/c/oxford-102-flower-pytorch/
Private score is considered for accuracy.
