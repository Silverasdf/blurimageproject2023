# Blur Image Training Project

This is a project I had done over my 10-week internship at ORNL. The goal of this project was to use different Computer Vision Techniques to classify pictures of blurred car cabin images

I did not upload the models or results here, but feel free to run them yourself off of your own data. Paper coming soon...

## Different techniques used

### Image Classification: ResNet18, EfficientNetB7, Vision Transformer B 16, CLIP Zero-Shot, CLIP Fine-Tuned

### Object Detection: Faster-RCNN Face Detection: RetinaFace

## Folders

### Config: Contains a template for a config file for the different programs

### src: Contains the source code for the different programs

### Analysis: Contains Code I ran for the plots

## src

### cleanup.py: Contains a lot of helper functions for the other programs

### clipzeroshot.py: Contains the code for the CLIP Zero-Shot method. Outputs a results JSON file

### clipfinetune.py: Contains the code for the CLIP Fine-Tuned method. Requires a config file. Outputs a results JSON file

### facedetectortest.py: Contains the code for the Face Detection method. Writes detection files to a folder

### mydms.py: Contains Data Modules I used for the ResNet18, EfficientNetB7, Vision Transformer B 16 methods

### mymodels.py: Contains the models I used for the ResNet18, EfficientNetB7, Vision Transformer B 16 methods

### pl_trainevaltestsave.py: Driver code for the ResNet18, EfficientNetB7, Vision Transformer B 16 methods. Requires a config file

### Object_Detector_Test.py: Takes from the detection file and Ground Truth CSV to create a results JSON file

### run.sh: Used to run pl_trainevaltestsave.py for multiple config files, if that is needed

### run_model.py: Used to run one model against testing data. Settings are all on the commandline and more details can be found in the header

Note: Read headers for more details on each program. Some of these programs are based on other programs from other repositories. I have included the links to the original repositories in the headers of the programs

## Config

### clip_config.py: Config file for CLIP Fine-Tuned

### pl_config.py: Config file for ResNet18, EfficientNetB7, Vision Transformer B 16
