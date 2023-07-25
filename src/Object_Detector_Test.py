#Object Detector Test - Ryan Peruski, 06/28/2023
#Tests the object detector by comparing the detections to the ground truth
import os
import pandas as pd
from cleanup import iou
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score

gt = pd.read_csv('/root/BlurImageTrainingProject/Analysis/Analysis_All/MasterSheet_2023_updated.csv')
det_dir = '/root/BlurImageTrainingProject/Experiments/Retina_Detections'
output_dir = '/root/BlurImageTrainingProject/Experiments/Retina_Results'
model_type = "RetinaFace"
back_res = [91, 91]
front_res = [275, 320]
picture_size = [720, 480]

front_pred=[]
back_pred=[]
front_scores=[]
back_scores=[]
front_gt=[]
back_gt=[]
front_names=[]
back_names=[]
#Go through every row
for index, row in gt.iterrows():
    #Find the corresponding detection file
    det_file = os.path.join(det_dir, row['Filename'][:-4] + ".txt")
    #Open the detection file
    dets = []
    guesses = [0] * 8

    with open(det_file, 'r') as f:
        for line in f:
            dets.append(line.strip().split(' '))
    for det in dets:
        ious = []
        guessbox = [int(det[2]), int(det[3]), int(det[4]), int(det[5])]
        score = float(det[1])

        #Driver
        x1 = max(row["DriverX"] - (front_res[0] / 2), 0)
        y1 = max(row["DriverY"] - (front_res[1] / 2), 0)
        x2 = min(row["DriverX"] + (front_res[0] / 2), picture_size[0])
        y2 = min(row["DriverY"] + (front_res[1] / 2), picture_size[1])
        actualbox = [x1, y1, x2, y2]  
        ious.append(iou(guessbox, actualbox))

        #Passenger
        x1 = max(row["PassengerX"] - (front_res[0] / 2), 0)
        y1 = max(row["PassengerY"] - (front_res[1] / 2), 0)
        x2 = min(row["PassengerX"] + (front_res[0] / 2), picture_size[0])
        y2 = min(row["PassengerY"] + (front_res[1] / 2), picture_size[1])
        actualbox = [x1, y1, x2, y2]
        ious.append(iou(guessbox, actualbox))

        #Backseat
        for i in range(6):
            x1 = max(row[f"RearSeatX{i+1}"] - (back_res[0] / 2), 0)
            y1 = max(row[f"RearSeatY{i+1}"] - (back_res[1] / 2), 0)
            x2 = min(row[f"RearSeatX{i+1}"] + (back_res[0] / 2), picture_size[0])
            y2 = min(row[f"RearSeatY{i+1}"] + (back_res[1] / 2), picture_size[1])
            actualbox = [x1, y1, x2, y2]
            ious.append(iou(guessbox, actualbox))

        #Find the highest iou
        max_iou = max(ious)
        max_index = ious.index(max_iou)
        guesses[max_index] = score
        # print(det, ious)
    actual = [int(row["DriverOcc"]), int(row["PassengerOcc"]), row["Occ1"], row["Occ2"], row["Occ3"], row["Occ4"], row["Occ5"], row["Occ6"]]
    for i, s in enumerate(actual):
        if i > 1: #Back Seat
            if s != -1:
                back_scores.append(guesses[i])
                back_pred.append(1 if guesses[i] > 0.5 else 0)
                back_gt.append(s)
                back_names.append(row['Filename'])

        else: #Front Seat
            if s != -1:
                front_scores.append(guesses[i])
                front_pred.append(1 if guesses[i] > 0.5 else 0)
                front_gt.append(s)
                front_names.append(row['Filename'])
    # print(front_gt, front_pred, front_scores)
    # print(back_gt, back_pred, back_scores)
    # print(dets, guesses, actual)
# print(len(front_gt), len(front_pred), len(front_scores))
# print(len(back_gt), len(back_pred), len(back_scores))


#From here on it's plotting

predictions = np.array(front_pred)
targets = np.array(front_gt)
scores = np.array(front_scores)
names = np.array(front_names)
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

prc_path = os.path.join(output_dir, f'{model_type}_0_precision_recall_curve.png')
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
plt.xticks(np.arange(2))
plt.yticks(np.arange(2))

# Add value annotations to the confusion matrix cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")

cm_path = os.path.join(output_dir, f'{model_type}_0_confusion_matrix.png')
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
roc_path = os.path.join(output_dir, f'{model_type}_0_roc_curve.png')
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
text = f"Acc: {100*acc:.2f}%\nType: {model_type}"
draw.text((10, 10), text, fill='black', font=text_font)

# Save the combined image
combined_image_path = os.path.join(output_dir, f'{model_type}_front_perfs.png')
combined_image.save(combined_image_path)

#Remove the individual images
for path in image_paths:
    os.remove(path)

df = pd.DataFrame({'name': names, 'y_true': targets, 'y_pred': predictions, 'y_scores': scores, 'mode': model_type, 'prevalence': prevalence})
df.to_json(os.path.join(output_dir, f'{model_type}_front_perfs.json'))

#Back Seat

predictions = np.array(back_pred)
targets = np.array(back_gt)
scores = np.array(back_scores)
names = np.array(back_names)

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

prc_path = os.path.join(output_dir, f'{model_type}_0_precision_recall_curve.png')
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
plt.xticks(np.arange(2))
plt.yticks(np.arange(2))

# Add value annotations to the confusion matrix cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")

cm_path = os.path.join(output_dir, f'{model_type}_0_confusion_matrix.png')
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
roc_path = os.path.join(output_dir, f'{model_type}_0_roc_curve.png')
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
text = f"Acc: {100*acc:.2f}%\nType: {model_type}"
draw.text((10, 10), text, fill='black', font=text_font)

# Save the combined image
combined_image_path = os.path.join(output_dir, f'{model_type}_back_perfs.png')
combined_image.save(combined_image_path)

#Remove the individual images
for path in image_paths:
    os.remove(path)

df = pd.DataFrame({'name': names, 'y_true': targets, 'y_pred': predictions, 'y_scores': scores, 'mode': model_type, 'prevalence': prevalence})
df.to_json(os.path.join(output_dir, f'{model_type}_back_perfs.json'))