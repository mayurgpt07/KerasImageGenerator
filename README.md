# KerasImageGenerator
[Kaggle Problem](https://www.kaggle.com/c/dog-breed-identification/overview) - Identify dog breed using the images provided <br />
Code: Use Keras image generator to load the data and train a Convolution Neural Network (CNN) <br />
Usecase: Automatically read images and categories without explicitly converting each image into an array of data points 

## Requirements:
1) Tensorflow 2.0 
2) [Data](https://www.kaggle.com/c/dog-breed-identification/data) Folders - Train and Test data folders must be present
3) labels.csv - Mapping of image id and category (dog breed)

## Steps: 
1) Run DogBreed.py - Creates a new directory structure with images images of similar breed in a single folder
2) Run DogBreedClassification.py - Trains a CNN for dog breed identification

### Note:
The code is not written to achieve high accuracy, it is just an introduction on how to use keras image generator. To achieve good performance you can eithe use Resnet (code in DogBreedClassification.py) or modify the custom model 

