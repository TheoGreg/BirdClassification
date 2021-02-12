# BirdClassification

Bird classification model based on Reliable Majority Voting (Resnext, Resnet, VGG, Densenet, Efficientnet) in top 12% of MVA RecVis Kaggle Challenge.

## Dataset
It is possible to download the bird dataset [here](https://drive.google.com/file/d/1GIYYPXfoXcrRup6rtpxW1Cvz9rqm9kx_/view?usp=sharing).

## Method
As a first step, Faster-RCNN is applied to the dataset in order to crop images and homogenize the dataset.

![alt text](https://github.com/TheoGreg/BirdClassification/blob/main/results/FasterRCNN_cropping.png)

The model is a form of majority voting between Resnext, Resnet, VGG, Densenet and Efficientnet classificators. 
Reliable Majority Voting (RMV) method was mainly used to enhance robustness thanks to an ensemble model. The concept is a voting system between classificators whose votes are weighted by their train accuracy on their predicted classes.

![alt text](https://github.com/TheoGreg/BirdClassification/blob/main/results/model_validation_accuracy.png)

For more details, you can access the one-page report available on the repository.
