import os
import shutil
import random
import numpy as np

# Directory to get data from
dir = r'*** put directory path here ***'
# Directory to store training set
train_dir = r'*** put directory path here ***'
# Directory to store testing set
test_dir = r'*** put directory path here ***'
# Directory to store validation path
valid_dir = r'*** put directory path here ***'

# Get the list of file paths in the directory of dataset
files = [file for file in os.listdir(
    dir) if os.path.isfile(os.path.join(dir, file))]
# Input size of training set
train_count = np.round(70 / 100 * len(files))
# Input size of testing set
test_count = np.round(15 / 100 * len(files))
# Input size of validation set
valid_count = np.round(15 / 100 * len(files))
# Generate random numbers of file indices
random_indices = list(random.sample(range(0, len(files)), len(files)))
print("len(files)", len(files))

# train_files indices
print(random_indices)

# training files
train_file_index = random_indices[0:int(train_count) + 1]
train_file_name = [files[i] for i in train_file_index]

# testing files
test_file_index = random_indices[int(
    train_count) + 1:int(train_count + test_count) + 1]
test_file_name = [files[i] for i in test_file_index]

# validation files
valid_file_index = random_indices[int(train_count + test_count) + 1:]
valid_file_name = [files[i] for i in valid_file_index]

# training files
for train in train_file_name:
    file = train
    shutil.copyfile(os.path.join(dir, file), os.path.join(train_dir, file))
# test_files
for test in test_file_name:
    file = test
    shutil.copyfile(os.path.join(dir, file), os.path.join(test_dir, file))

# valid_files
for valid in valid_file_name:
    file = valid
    shutil.copyfile(os.path.join(dir, file), os.path.join(valid_dir, file))
