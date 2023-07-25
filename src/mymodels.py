from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as metrics
import torchvision.models as models
from torchmetrics.functional import accuracy
from efficientnet_pytorch import EfficientNet
import timm
import lightning as L
import os, sys
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from torchvision.models import vision_transformer as vits
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import clip
import torch.nn.utils.rnn as rnn_utils
from mydms import ImageData, ImageDataTest
from transformers import AutoModel, PretrainedConfig, AutoConfig

# Normal Lightning Model, but supports EfficientNetB7 and ResNet18
class LitModel(L.LightningModule):
    def __init__(self, lr=0.01, num_classes=2, load=[False, 0], model_dir='.', model="EfficientNetB7"):
        super().__init__()
        self.num_classes = num_classes

        self.save_hyperparameters()
        if model=="EfficientNetB7":
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
            num_features = self.model._fc.in_features
            self.model._fc = nn.Linear(num_features, self.num_classes)
        elif model=="ResNet18":
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        else:
            raise Exception(f"mymodels.py: Model {model} not implemented")
        
        self.model_type = model
        self.model_num = load[1]
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Load model if specified. If the model doesn't exist, exit.
        if load[0]:
            try:
                test = torch.load(f'{model_dir}/new_model_{load[1]}.pth')
                self.model = test
            except Exception as e:
                print(f"mymodels.py: Error loading model: {e}")
                sys.exit(1)



    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        #print(preds," before")
        preds %= self.num_classes
        #print(preds," after")
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        #print(preds," before")
        preds %= self.num_classes
        #print(preds, " after")
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
class LitModelSave(LitModel):
    def __init__(self, lr=0.01, num_classes=2, load=[False, 0], model_dir='.', output_dir='.', model="EfficientNetB7",mode='Unknown',test_data_dir='.'):
        super().__init__(lr, num_classes, load, model_dir, model)
        self.save_hyperparameters()
        self.mode = mode
        self.output_dir = output_dir
        self.test_data_dir = test_data_dir

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        #print(preds," before")
        preds %= 2
        #print(preds, " after")
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        # Store predictions and targets for later use
        self.predictions.append(preds.detach().cpu().numpy())
        self.targets.append(y.detach().cpu().numpy())
        self.scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        return loss
    
    # Plots the performance curves and saves to the results directory
    def on_test_end(self):
        # Put together predictions and targets
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        scores = np.concatenate(self.scores)
        self.test_dataset=ImageData(self.test_data_dir)
        self.test_dataset.setup()
        filenames = self.test_dataset.return_test_filenames()
        filenames = [os.path.basename(f) for f in filenames]

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, scores)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        # Plot Prevalence Threshold
        prevalence = np.sum(targets) / len(targets)
        plt.axhline(y=prevalence, color='r', linestyle='--', label='Prevalence Threshold')
        plt.legend()

        prc_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_precision_recall_curve.png')
        plt.savefig(prc_path)
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(np.arange(self.num_classes))
        plt.yticks(np.arange(self.num_classes))

        # Add value annotations to the confusion matrix cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        cm_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = roc_auc_score(targets, predictions)
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
        plt.grid(True)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guessing') #Plot y = x line
        plt.legend()
        roc_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()

        df = pd.DataFrame({'name': filenames, 'y_true': targets, 'y_pred': predictions, 'y_scores': scores, 'mode': self.mode, 'prevalence': prevalence})
        df.to_json(os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_perfs.json'))

        # Combine images into one picture with annotations
        image_paths = [prc_path, cm_path, roc_path]
        images = [Image.open(path) for path in image_paths]
        widths, heights = zip(*(img.size for img in images))

        # Determine the dimensions of the combined image
        combined_width = sum(widths)
        combined_height = max(heights)

        # Create the blank combined image
        combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

        # Paste the individual images into the combined image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Add text annotations to the combined image
        draw = ImageDraw.Draw(combined_image)
        text_font = ImageFont.load_default()
        acc = accuracy_score(targets, predictions)
        text = f"Acc: {100*acc:.2f}%\nType: {self.mode}"
        draw.text((10, 10), text, fill='black', font=text_font)

        # Save the combined image
        combined_image_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_perfs.png')
        combined_image.save(combined_image_path)

        #Remove the individual images
        for path in image_paths:
            os.remove(path)


class ViTClassifier(L.LightningModule):

    def __init__(self, num_classes=2, lr=1e-2, load=[False, 0], model_dir='.', model='ViT'):
        super(ViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = lr

        self.save_hyperparameters()
        if model == 'ViT':
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        else:
            print("mymodels.py: Error: Invalid model type")
            sys.exit(1)

        self.criterion = nn.CrossEntropyLoss()

        self.model_type = model
        self.model_num = load[1]

        # Load model if specified. If the model doesn't exist, exit.
        if load[0]:
            try:
                test = torch.load(f'{model_dir}/new_model_{load[1]}.pth')
                self.vit = test
            except Exception as e:
                print(f"mymodels.py: Error loading model: {e}")
                sys.exit(1)

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
class ViTClassifierSave(ViTClassifier):
    def __init__(self, lr=0.01, num_classes=2, load=[False, 0], model_dir='.', output_dir='.', model="ViT",mode='Unknown',test_data_dir='.'):
        super().__init__(num_classes, lr, load, model_dir, model)
        self.save_hyperparameters()
        self.mode = mode
        self.output_dir = output_dir
        self.test_data_dir = test_data_dir
        self.predictions = []
        self.targets = []
        self.scores = []
        self.names = []

    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     #Include each of the filenames in the batch
    #     print("mymodels.py: test_dataloader: Loading test dataset")
    #     self.filenames = self.test_dataset.filenames
    #     return super().test_dataloader()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        # Store predictions and targets for later use
        self.predictions.append(preds.detach().cpu().numpy())
        self.targets.append(y.detach().cpu().numpy())
        self.scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return loss
    
    # Plots the performance curves and saves to the results directory
    def on_test_end(self):
        self.test_dataset=ImageData(self.test_data_dir)
        self.test_dataset.setup()
        filenames = self.test_dataset.return_test_filenames()
        filenames = [os.path.basename(f) for f in filenames]
        # Put together predictions and targets
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        scores = np.concatenate(self.scores)
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, scores)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        # Plot Prevalence Threshold
        prevalence = np.sum(targets) / len(targets)
        plt.axhline(y=prevalence, color='r', linestyle='--', label='Prevalence Threshold')
        plt.legend()

        prc_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_precision_recall_curve.png')
        plt.savefig(prc_path)
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(np.arange(self.num_classes))
        plt.yticks(np.arange(self.num_classes))

        # Add value annotations to the confusion matrix cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        cm_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = roc_auc_score(targets, predictions)
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (AUC = {:.2f})'.format(roc_auc))
        plt.grid(True)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guessing') #Plot y = x line
        plt.legend()
        roc_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()

        # Combine images into one picture with annotations
        image_paths = [prc_path, cm_path, roc_path]
        images = [Image.open(path) for path in image_paths]
        widths, heights = zip(*(img.size for img in images))

        # Determine the dimensions of the combined image
        combined_width = sum(widths)
        combined_height = max(heights)

        # Create the blank combined image
        combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

        # Paste the individual images into the combined image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Add text annotations to the combined image
        draw = ImageDraw.Draw(combined_image)
        text_font = ImageFont.load_default()
        acc = accuracy_score(targets, predictions)
        text = f"Acc: {100*acc:.2f}%\nType: {self.mode}"
        draw.text((10, 10), text, fill='black', font=text_font)

        # Save the combined image
        combined_image_path = os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_perfs.png')
        combined_image.save(combined_image_path)

        #Remove the individual images
        for path in image_paths:
            os.remove(path)

        df = pd.DataFrame({'name': filenames, 'y_true': targets, 'y_pred': predictions, 'y_scores': scores, 'mode': self.mode, 'prevalence': prevalence})
        df.to_json(os.path.join(self.output_dir, f'{self.model_type}_{self.model_num}_perfs.json'))

class Load_Model(L.LightningModule):
    def __init__(self, num_classes=2, model_path='.', test_data_dir='.'):
        super().__init__()
        self.num_classes = num_classes
        self.test_data_dir = test_data_dir
        self.predictions = []
        self.targets = []
        self.scores = []

        self.save_hyperparameters()
        
        self.criterion = nn.CrossEntropyLoss()

        # Load model if specified. If the model doesn't exist, exit.
        try:
            test = torch.load(model_path)
            self.model = test
        except Exception as e:
            print(f"mymodels.py: Error loading model: {e}")
            sys.exit(1)

    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        #print(preds," before")
        preds %= self.num_classes
        #print(preds, " after")
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.predictions.append(preds.detach().cpu().numpy())
        self.targets.append(y.detach().cpu().numpy())
        self.scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return loss
    
    def on_test_end(self):
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        scores = np.concatenate(self.scores)
        self.test_dataset=ImageDataTest(self.test_data_dir)
        self.test_dataset.setup()
        filenames = self.test_dataset.return_test_filenames()
        filenames = [os.path.basename(f) for f in filenames]
        df = pd.DataFrame({'name': filenames, 'y_true': targets, 'y_pred': predictions, 'y_scores': scores})
        for i, row in df.iterrows():
            print(f"{row['name']}: {row['y_true']} -> {row['y_pred']} ({row['y_scores']})")