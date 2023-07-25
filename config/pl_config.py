# Config file for front seat part - See pl_trainevaltestsave.py for more details

import os
#Mode - used when naming the perf curve png files and on the json files
MODE = 'CNN FrontSeat'

#Location of the ground truth images - used for redisperse
DISPERSE_LOCATION = '/root/BlurImageTrainingProject/Data_Front/TrainingAndValidation' #For old data

#Training Data - must have subdirectories for Training, Validation, and Testing, The testing subdirectory can just have dummy images, since it isn't meant to be used
OLD_DIR = '/root/BlurImageTrainingProject/Data_Front/New_Data'

#Testing Data - must have subdirectories for Training, Validation, and Testing. The training and validation subdirectories can just have dummy images, since they aren't meant to be used
NEW_DIR = '/root/BlurImageTrainingProject/Data_Front/New_Data_2023'

#Where models are both saved and loaded from
MODEL_DIR = '/root/BlurImageTrainingProject/Experiments/EfficientNet_Front_Models'

#Where results are saved - these are saved into JSON files and also saved as png files
SAVE_DIR = '/root/BlurImageTrainingProject/Experiments/EfficientNet_Front_Results'

#Model type - must be one of the following: "ResNet18", "EfficientNetB7", "ViT"
MODEL_TYPE = "EfficientNetB7"

#Flag for whether to train the model - set to false to skip directly to testing
TRAIN = False

#Flag for whether to save the model - set to false to not save the model nor the results
SAVE = False

#Flag for whether to redisperse the data - set to false to skip redisperse
REDISPERSE = False
REDISPERSE_FREQUENCY = 10 #After how many models does the data redisperse? Can be ignored if REDISPERSE is set to false

#Number of models to train/test (depending on whether the TRAIN flag is set to true or false)
NUM_OF_MODELS = len(os.listdir(MODEL_DIR))

#Hyperparameters
BATCH_SIZE = 4
EPOCHS = 1000
PATIENCE = 10
lr = 0.01