# Run Model - All this does is load a model, load testing data, and run the model on the testing data
# Ryan Peruski, 07/25/23
# usage: python run_model.py <model_path> <data_dir>
# Note that the results are printed to stdout

import os, sys
import lightning as L
import torch
import warnings
import importlib.util
warnings.filterwarnings("ignore", ".*does not have many workers.*")

#My modules
from mydms import ImageDataTest

#Parse args
if len(sys.argv) < 3:
    print("usage: python run_model.py <model_path> <data_dir>")
    exit(1)

model_dir = sys.argv[1]
data_dir = sys.argv[2]

dm_2023 = ImageDataTest(data_dir=data_dir, batch_size=4)

model = torch.load(model_dir)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=[0],
)

trainer.test(model, dm_2023)