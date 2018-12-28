# pytorch_challenge

## Getting Started

...

## Rubric

### Files Submitted

- [x] Submission Files, The submission includes all required files. (Model checkpoints not required.)

### Part 1 - Development Notebook

- [x] Package Imports, All the necessary packages and modules are imported in the first cell of the notebook
- [x] Training data augmentation, torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
- [x] Data normalization, The training, validation, and testing data is appropriately cropped and normalized
- [x] Data batching, The data for each set is loaded with torchvision's DataLoader
- [x] Data loading, The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
- [x] Pretrained Network, A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
- [x] Feedforward Classifier, A new feedforward network is defined for use as a classifier using the features as input
- [x] Training the network, The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
- [ ] Testing Accuracy, The network's accuracy is measured on the test data
- [x] Validation Loss and Accuracy, During training, the validation loss and accuracy are displayed
- [x] Loading checkpoints, There is a function that successfully loads a checkpoint and rebuilds the model
- [x] Saving the model, The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
- [x] Rubric, Image Processing, The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
- [x] Class Prediction, The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image
- [x] Sanity Checking with matplotlib, A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

### Part 2 - Command Line Application

- [ ] Training a network, train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
- [ ] Training validation log, The training loss, validation loss, and validation accuracy are printed out as a network trains
- [ ] Model architecture, The training script allows users to choose from at least two different architectures available from torchvision.models
- [ ] Model hyperparameters, The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
- [ ] Training with GPU, The training script allows users to choose training the model on a GPU
- [ ] Predicting classes, The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
- [ ] Top K classes, The predict.py script allows users to print out the top K classes along with associated probabilities
- [ ] Displaying class names, The predict.py script allows users to load a JSON file that maps the class values to other category names
- [ ] Predicting with GPU, The predict.py script allows users to use the GPU to calculate the predictions

### Misc

- [ ] Add `requirements.txt`
- [ ] Complete "Getting Started" section

## Reference

### PyTorch
- https://pytorch.org/
- https://pytorch.org/tutorials/
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html
- https://pytorch.org/docs/stable/data.html
- https://pytorch.org/docs/stable/nn.html
- https://pytorch.org/docs/stable/torchvision/transforms.html
- https://pytorch.org/docs/stable/torchvision/models.html

### Example Code
- https://github.com/udacity/deep-learning-v2-pytorch
- https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/cifar-cnn/cifar10_cnn_solution.ipynb
- https://github.com/udacity/pytorch_challenge
- https://github.com/wanderdust/flower_classification
- https://github.com/Mdhvince/CNN_FlowerClassification
- https://github.com/Alihussain1/Flower-Classification, Count of flowers in each category
- https://github.com/rajesh-iiith/AIPND-ImageClassifier
- https://github.com/debajyotiguha11/Flower-Classification
- https://github.com/nirajpandkar/flowers-classification-pytorch
- https://github.com/momothepikachu/pytorch-flower-classification
- https://github.com/GabrielePicco/deep-learning-flower-identifier
- https://github.com/twishasaraiya/PyTorch-Scholarship-Lab-Challenge
- https://github.com/vprayagala/Udacity-Facebook-Scholarship-Challenge
- https://github.com/miguelangel/ai--transfer-learning-for-image-classification

### Others
- https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
- [PyTorch Facebook Project Public Leaderboard](https://docs.google.com/spreadsheets/d/1eVqdzQtS4xJDO-nZB8E3PvhpSgYML5dR7Mdh5CCtt-E/edit#gid=0)
- [Udacity PyTorch Lab Challenge](https://docs.google.com/document/d/1-MCDPOejsn2hq9EoBzMpzGv9jEdtMWoIwjkAa1cVbSM/edit#)
