# Face Detector Test - This writes detection files for the RetinaFace model
# Ryan Peruski, 07/25/23
# I did not too much to this program. Original can be found here:
# https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/
# pip install retina-face
# https://pypi.org/project/retina-face/

from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

data_dir='F:/BlurImageTrainingProject/FrontSeatPart/Data/RawData2023'
save_dir='F:/BlurImageTrainingProject/retinaface/Results'
ann_dir = 'F:/BlurImageTrainingProject/retinaface/RetinaAnnotations'

for num, img_path in enumerate(os.listdir(data_dir)):
  print(f'Image {num} of {len(os.listdir(data_dir))}: {os.path.basename(img_path)}')
  df = pd.DataFrame(columns=['img_path', 'faceX', 'faceY', 'Score', 'BB1', 'BB2', 'BB3', 'BB4'])
  img_path = os.path.join(data_dir, img_path)
  if not img_path.endswith('.jpg'):
    continue

  faces = RetinaFace.detect_faces(img_path)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for face in faces:
    print(face)
    if type(face) != str: # if face is not detected
      continue
    score=faces[face]['score']
    facial_area = faces[face]['facial_area']
    landmarks = faces[face]['landmarks']
    RE = landmarks['right_eye']
    LE = landmarks['left_eye']
    NOSE = landmarks['nose']
    RMOUTH = landmarks['mouth_right']
    LMOUTH = landmarks['mouth_left']
  
    #highlight facial area
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
    #add to the df
    #Grab centroids of the facial area
    centerX = (facial_area[2] + facial_area[0])/2
    centerY = (facial_area[3] + facial_area[1])/2
    df.loc[len(df)] = [face, centerX, centerY, score, facial_area[0], facial_area[1], facial_area[2], facial_area[3]]
    #plt.imshow(img)
    #plt.show()
    #Save the image

    #extract facial area
    # img = cv2.imread(img_path)
    # facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
  
    #highlight the landmarks
    cv2.circle(img, tuple((round(landmarks["left_eye"][0]),round(landmarks["left_eye"][1]))), 6, (0, 0, 255), -1)
    cv2.circle(img, tuple((round(landmarks["right_eye"][0]),round(landmarks["right_eye"][1]))), 6, (0, 0, 255), -1)
    cv2.circle(img, tuple((round(landmarks["nose"][0]),round(landmarks["nose"][1]))), 1, (0, 0, 255), -1)
    cv2.circle(img, tuple((round(landmarks["mouth_left"][0]),round(landmarks["mouth_left"][1]))), 1, (0, 0, 255), -1)
    cv2.circle(img, tuple((round(landmarks["mouth_right"][0]),round(landmarks["mouth_right"][1]))), 1, (0, 0, 255), -1)

    #plt.imshow(img)
    #plt.show()
  df.to_csv(os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '.csv')), index=False)
  cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

  #Write annotation to annotation file
  with open(os.path.join(ann_dir, os.path.basename(img_path).replace('.jpg', '.txt')), 'w') as f:
    for index, row in df.iterrows():
      f.write(f"person {row['Score']} {row['BB1']} {row['BB2']} {row['BB3']} {row['BB4']}\n")


  # faces = RetinaFace.extract_faces(img)
  # for face in faces:
  # plt.imshow(face)
  # plt.show()
