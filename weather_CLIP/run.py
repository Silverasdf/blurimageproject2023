# Run.py - Takes code from clipzeroshot and modifies it for weather data
# Ryan Peruski, 07/21/23
# usage: python run.py [-verbose]
import torch
from PIL import Image
import clip
import os, sys
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

#Some config here
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dir = "/root/BlurImageTrainingProject/new_weather_pictures_for_ornl"
output_file = '/root/BlurImageTrainingProject/weather_CLIP/results.json'
batch_size = 4
classes = ['clear', 'cloudy', 'fog', 'night', 'partly_cloudy', 'rain', 'snow'] # Actual names of classes
# text_inputs = ["view of a road in clear weather", "view of a road in cloudy weather", "view of a road in foggy weather", "view of a road at night", "view of a road in partly cloudy weather", "view of a road in rainy weather", "view of a road in snowy weather"] #Prompt engineering is here
text_inputs = ["view of a road in clear weather", "view of a road in cloudy weather", "foggy", "view of a road at night", "view of a road in partly cloudy weather", "view of a road in rainy weather", "view of a road in snowy weather"] #Prompt engineering is here


# Parse args for verbose flag
verbose = False
if len(sys.argv) > 1:
    verbose = sys.argv[1] == '-verbose'

filenames = []
for root, dirs, files in os.walk(dir):
    for filename in files:
        # Prepare the image
        if not filename.endswith('.png'):
            continue
        filenames.append(os.path.join(os.path.basename(root), os.path.basename(filename)))

batches = math.ceil(len(filenames)/batch_size)

# Get the labels for the scores
labels = text_inputs

targets = []
predictions = []

#For each image, get the label with the highest score
pbar = tqdm(total=batches, desc="Processing batches", unit="batch")
for i in range(batches):
    #Take the batch of images
    images = []
    for j in range(batch_size):
        try:
            image_path = os.path.join(dir, filenames[i*batch_size + j])
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
        
        if verbose:
            print(f'{filenames[i*batch_size+j]} is {labels[scores.argmax()]}')
            print(f'Scores: {scores}')

        #Append
        name = filenames[i*batch_size+j].split('/')[0]
        index = classes.index(name)
        targets.append(index)

        index = scores.argmax()
        predictions.append(index)
    
if verbose:
    print(f'Targets: {targets}')
    print(f'Predictions: {predictions}')

df = pd.DataFrame({'y_true': targets, 'y_pred': predictions})
df.to_json(output_file)