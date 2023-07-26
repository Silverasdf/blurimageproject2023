# Run Model - All this does is load a model, load testing data, and run the model on the testing data
# Ryan Peruski, 07/25/23
# usage: python run_model.py <model_path> <data_dir> <output_file>
# Output file must be a .csv file
# Note that the results are printed to stdout output is as follows:
# {filename}: {label} -> {prediction} ({score for 0}, {score for 1})

import os, sys
import lightning as L
import torch
import warnings
import importlib.util
warnings.filterwarnings("ignore", ".*does not have many workers.*")

#My modules
from mymodels import Load_Model
from mydms import ImageDataTest

#Parse args
if len(sys.argv) < 4:
    print("usage: python run_model.py <model_path> <data_dir> <output_file>")
    sys.exit(1)

model_path = sys.argv[1]
data_dir = sys.argv[2]
output_file = sys.argv[3]

if not output_file.endswith('.csv'):
    print("output_file must end with .csv", file=sys.stderr)
    sys.exit(1)

dm_2023 = ImageDataTest(data_dir=data_dir, batch_size=4)

model = Load_Model( # This model inherits from LitModel and automatically saves perf curves. Use LitModel if you don't want to save perf curves
    num_classes=dm_2023.num_classes, 
    model_path=model_path,
    test_data_dir=data_dir,
    output_file=output_file
)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=[0],
)

trainer.test(model, dm_2023)