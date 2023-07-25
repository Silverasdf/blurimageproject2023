# Clipzeroshot.py - Used for zero-shot CLIP model testing
# Ryan Peruski, 07/21/23
# usage: python cliptest.py [-verbose]
import torch
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve
from PIL import Image
import clip
import os, sys
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm

# Small config section below
# Set to true for backseat, false for frontseat
back = False
save = False
text_inputs = ["Blurred Automobile interior with no one in the front passenger seat", "Car with a blurred person in the passenger seat"]
output_dir = '/root/BlurImageTrainingProject/Experiments/CLIP'
batch_size = 4

if back:
    dir = "/root/BlurImageTrainingProject/Data_Back/New_Data_2023/Testing/0" #No Person class
    dir2 = "/root/BlurImageTrainingProject/Data_Back/New_Data_2023/Testing/1" #Person class
    mode = 'CLIP - BackSeat' #This is exclusively for the mode section of the json file
    type = 'CLIP_Back' #This is what the results will be saved as -> {mode}_perfs.json
else:
    dir = "/root/BlurImageTrainingProject/Data_Front/New_Data_2023/Testing/0"
    dir2 = "/root/BlurImageTrainingProject/Data_Front/New_Data_2023/Testing/1"
    mode = 'CLIP - FrontSeat'
    type = 'CLIP_Front'

# Load the CLIP model
verbose = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# Parse args for verbose flag
if len(sys.argv) > 1:
    verbose = sys.argv[1] == '-verbose'

images = []
filenames = []
for num, filename in enumerate(os.listdir(dir)):
    # Prepare the image
    if not filename.endswith('.jpg'):
        continue
    filenames.append('0/'+filename)

for num, filename in enumerate(os.listdir(dir2)):
    # Prepare the image
    if not filename.endswith('.jpg'):
        continue
    filenames.append('1/'+filename)

batches = math.ceil(len(filenames)/batch_size)

# Get the labels for the scores
labels = text_inputs

all_scores = []
targets = []
predictions = []

# For each image, get the label with the highest score
tp = 0
fp = 0
tn = 0
fn = 0
pbar = tqdm(total=batches, desc="Processing batches", unit="batch")
for i in range(batches):
    #Take the batch of images
    images = []
    for j in range(batch_size):
        try:
            image_path = os.path.join(dir[:-2], filenames[i*batch_size + j])
        except IndexError:
            break
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image)
        images.append(image)

    pbar.update(1)
    # Tokenize the text inputs
    image_inputs = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(text_inputs).to(device)

    # Generate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_inputs).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity_scores = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)
    #Normalize scores
    similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())    


    #For each image, get the label with the highest score
    for j, scores in enumerate(similarity_scores.T):
        #Make sure the scores add up to 1
        scores = scores / scores.sum()
        if filenames[i*batch_size+j].endswith('driver.jpg'):
            continue
        
        if verbose:
            print(f'{filenames[i*batch_size+j]} is {labels[scores.argmax()]}')
            print(f'Scores: {scores}')
        if filenames[i*batch_size+j][0] == '1' and labels[scores.argmax()] == labels[1]:
            tp += 1
            predictions.append('1')
        elif filenames[i*batch_size+j][0] == '1' and labels[scores.argmax()] == labels[0]:
            fn += 1
            predictions.append('0')
        elif filenames[i*batch_size+j][0] == '0' and labels[scores.argmax()] == labels[1]:
            fp += 1
            predictions.append('1')
        elif filenames[i*batch_size+j][0] == '0' and labels[scores.argmax()] == labels[0]:
            tn += 1
            predictions.append('0')
        all_scores.append(scores[1])
        targets.append(filenames[i*batch_size+j][0])

    
if verbose:
    print(f'Scores: {all_scores}')
    print(f'Targets: {targets}')
    print(f'Predictions: {predictions}')

print(f'Labels: {labels}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Accuracy: {(tp+tn)/(tp+tn+fp+fn)}')
print(f'Precision: {tp/(tp+fp)}')
print(f'Recall: {tp/(tp+fn)}')
print(f'F1: {2*tp/(2*tp+fp+fn)}')

prevalence = (tp+fn)/(tp+tn+fp+fn)

# Save results
if save:
    df = pd.DataFrame({'name': filenames, 'y_true': targets, 'y_pred': predictions, 'y_scores': all_scores, 'mode': mode, 'prevalence': prevalence})
    df.to_json(os.path.join(output_dir, f'{type}_perfs.json'))