# Coloring Style Classification
Simple classifier to recognize the coloring style of input image from six categories as below.
![image](https://img09.deviantart.net/5460/i/2015/277/c/3/different_colouring_styles__by_k_shinobu-d9byqaw.png)
(different_colouring_styles__by_k_shinobu-d9byqaw)

## Dataset
The dataset was crawled from pixiv and clean manually.  
The total number of training and validation data are 4134 and 112, respectively.

## CNN architecture
Using pre-trained resnet-18 provided by torchvision.

## Train-val Curve Visualization in Tensorboard
![image](imgs/tensorboard.png)

## Confusion Matrix
![image](imgs/confusion_matrix.png)

