## Flower-Classifier
Classifier models trained on 102 Category flower dataset by Oxford

# DataSet
102 Category flower dataset released by Oxford. 
The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. 
In addition, there are categories that have large variations within the category and several very similar categories.

Original Dataset Website - > http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
I thank Professor Plum, Ph.D. for providing this dataset.

# Requirements
* Python >= 3.6
* PyTorch >= 1.0.0
* torchvision
* Pandas >=1.2.0

# Training
The following models were trained:
* AlexNet
* DenseNet201
* GoogLeNet
* MobileNet V2
* ResNet34

A linear layer was added to every model pretrained on ImageNet with dimension (no_in_features, no_of_classes), where no_in_features = 1000 and no_of_classes = 102.
# Results

| Model      | Accuracy |
|------------|----------|
|AlexNet     | 69.19%   |
|DenseNet201 | 84.35%   |
|GoogLeNet   | 83.12%   |
|MobileNet V2| 79.95%   |
|ResNet34    | 80.44%   |

Accuracy was obtained by submitting the submissions file to the kaggle competition - https://www.kaggle.com/c/oxford-102-flower-pytorch/
Private score is considered for accuracy.
