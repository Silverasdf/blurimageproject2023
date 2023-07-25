#Images for training set
image_path = '/root/BlurImageTrainingProject/Data_Front/New_Data/Training/1' #Person class
image_path2 = '/root/BlurImageTrainingProject/Data_Front/New_Data/Training/0'#No person class

#Images for testing set
test_image_path = '/root/BlurImageTrainingProject/Data_Front/New_Data_2023_edit/Testing/1' #Person class
test_image_path2 = '/root/BlurImageTrainingProject/Data_Front/New_Data_2023_edit/Testing/0' #No person class

#Model path - where models are both saved and loaded from
model_path = '/root/BlurImageTrainingProject/Experiments/CLIP_Front_Models'

#Results path - where results are saved - these are saved into JSON files
result_dir = '/root/BlurImageTrainingProject/Experiments/CLIP_Front_Results'

#Number of models to train
num_models = 4

#Device to use
import torch
device = "cuda:1" if torch.cuda.is_available() else "cpu"