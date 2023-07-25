# Clipfinetune from "Simple Implemtation" - Ryan Peruski, 07/01/2023
# This takes from a modification of the openai CLIP github and modifies it to work with the blur image dataset. Link to that specific GitHub Below
# https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/OpenAI%20CLIP%20Simple%20Implementation.ipynb
# usage: python clipfinetune.py <config_file>
import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
import albumentations as A
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
import sys
import importlib.util
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# Include argument for config file
# Example usage
if len(sys.argv) < 2:
    print("usage: python clipfinetune.py <config_file>")
    sys.exit(1)

config_file = sys.argv[1]
spec = importlib.util.spec_from_file_location("config", config_file)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
CONFIG = config_module

#Create model and save directories
if not os.path.exists(CONFIG.model_path):
    os.makedirs(CONFIG.model_path)
if not os.path.exists(CONFIG.result_dir):
    os.makedirs(CONFIG.result_dir)

#Create dataframe of all images and captions
df = pd.DataFrame(columns=['image', 'caption', 'id'])
dict_ = {'image': [], 'caption': []}
for file in os.listdir(CONFIG.image_path):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of a person")
for file in os.listdir(CONFIG.image_path2):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of an empty seat")
df['image'] = dict_['image']
df['caption'] = dict_['caption']
df['id'] = [id_ for id_ in range(len(df))]

df_test = pd.DataFrame(columns=['image', 'caption', 'id'])
dict_ = {'image': [], 'caption': []}
for file in os.listdir(CONFIG.test_image_path):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of a person")
for file in os.listdir(CONFIG.test_image_path2):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of an empty seat")
df_test['image'] = dict_['image']
df_test['caption'] = dict_['caption']
df_test['id'] = [id_ for id_ in range(len(df_test))]

#Config
class CFG:
    debug = False
    result_dir = CONFIG.result_dir
    image_path = CONFIG.image_path
    image_path2 = CONFIG.image_path2
    image_path_test = CONFIG.test_image_path
    image_path_test2 = CONFIG.test_image_path2
    model_path = CONFIG.model_path
    captions = df
    captions_test = df_test
    batch_size = 4
    num_workers = 1
    head_lr = 1e-5
    num_models=CONFIG.num_models
    image_encoder_lr = 1e-5
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 10
    factor = 0.8
    epochs = 1000
    train = False
    save = False
    device = CONFIG.device
    #model_name = 'resnet50'
    model_name = 'vit_base_patch16_224'
    image_embedding = 768 #2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

#Utils
class AvgMeter: #Used for the and progress bar
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        #Sorry for bad programming
        if os.path.exists(f"{CFG.image_path}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        elif os.path.exists(f"{CFG.image_path2}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path2}/{self.image_filenames[idx]}")
        elif os.path.exists(f"{CFG.image_path_test}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path_test}/{self.image_filenames[idx]}")
        else:
            image = cv2.imread(f"{CFG.image_path_test2}/{self.image_filenames[idx]}")

        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

#Used in the CLIP dataset
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.CenterCrop(224, 224, always_apply=True),
                A.Flip(),
                A.Affine(rotate=0, translate_percent=(0.025, 0.025)),
                A.ColorJitter(brightness=0.2, contrast=0.2), 
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.CenterCrop(224, 224, always_apply=True),
                A.Flip(),
                A.Affine(rotate=0, translate_percent=(0.025, 0.025)),
                A.ColorJitter(brightness=0.2, contrast=0.2),
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )

#Image Encoder - ViT B 16, pretrained
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

#Text Encoder - DistiliBERT, pretrained
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

#Processes imgs and text further after encoding
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    def get_sim(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        return logits


#Used in CLIPModel
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

#Helper function
def save(model, i):
    model_num = i
    torch.save(model, CONFIG.model_path + '/new_model_' + str(model_num) + '.pth')

#These each have their own image filenames, labels, etc.
def make_train_valid_dfs(caption=CFG.captions, valid_value=0.2):
    dataframe = caption.copy()
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(valid_value * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def traineval(model_num=0):
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            save(model, model_num)
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = torch.load(model_path).to(CFG.device)
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, num, n=9, maxi=0):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    labels = []
    preds = []
    num_ones =0
    for file in image_filenames:
        if os.path.exists(f'{CFG.image_path}/{file}') or os.path.exists(f'{CONFIG.test_image_path}/{file}'):
            labels.append(1)
            num_ones += 1
        else:
            labels.append(0)
    scores = [score[0] for score in dot_similarity.T.detach().cpu().numpy()]
    if maxi == 0:
        maxi = max(scores)
    #Normalize scores to be between 0 and 1
    new_scores = [score/maxi for score in scores]
    #Get best f1 score
    precision, recall, thres = precision_recall_curve(labels, new_scores)
    #List of f1 scores for each threshold
    f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(precision))]
    best_thresh = np.array(thres)[np.nanargmax(f1_scores)]
    best_f1 = max(f1_scores)
    print("Best F1: ", best_f1, "Best Threshold: ", best_thresh)
    #Normalize scores to be between 0 and 1
    for score in new_scores:
        if score >= best_thresh:
            preds.append(1)
        else:
            preds.append(0)
    print("Accuracy: ", accuracy_score(labels, preds))
    if CFG.save:
        df = pd.DataFrame({'name': image_filenames, 'y_true': labels, 'y_pred': preds, 'y_scores': new_scores, 'mode': "CLIP", 'prevalence': num_ones/len(image_filenames)})
        df.to_json(os.path.join(CFG.result_dir, f'CLIP_{num}_perfs.json'))

    # values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    # matches = [image_filenames[idx] for idx in indices[::5]]

    # _, axes = plt.subplots(3, 3, figsize=(10, 10))
    # for match, ax in zip(matches, axes.flatten()):
    #     if os.path.exists(f"{CFG.image_path}/{match}"):
    #         image = cv2.imread(f"{CFG.image_path}/{match}")
    #     else:
    #         image = cv2.imread(f"{CFG.image_path2}/{match}")
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ax.imshow(image)
    #     ax.axis("off")

    # plt.show()
    return maxi

#Main loop
for i in range(CFG.num_models):
    if not CFG.train and not os.path.exists(os.path.join(CFG.model_path, f"new_model_{i}.pth")):
        print(f"Model {i} does not exist")
        continue
    if CFG.train:
        print(f"Training Model {i}")
        traineval(i)
    model_path = os.path.join(CFG.model_path, f"new_model_{i}.pth")
    _, valid_df = make_train_valid_dfs(caption=CFG.captions, valid_value=0.15)
    model, image_embeddings = get_image_embeddings(valid_df, model_path=model_path)
    maxi = find_matches(model,
             image_embeddings,
             query="a picture of a person",
             image_filenames=valid_df['image'].values,
             num=i,
             n=9)
    _, valid_df = make_train_valid_dfs(caption=CFG.captions_test,valid_value=1)
    model, image_embeddings = get_image_embeddings(valid_df, model_path=model_path)
    find_matches(model,
             image_embeddings,
             query="a picture of a person",
             image_filenames=valid_df['image'].values,
             num=i,
             n=9,
             maxi=maxi)
