# Triplet Based Method
This folder contains the code for triplet based on-the-fly side-channel attack method. In this file, it explains what function that each source code file have and what envrionment that the code can run on, and also give detailed the command line of train and test the code.

## Content
This instructions for each python source code file is list below:

train.py: This is the source code for training the on-the-fly side-channel attack model
test.py: This is the source code for testing the on-the-fly side-channel attack model


## Data
For the data download, please refer the README.md at the root folder

## Requirement
This project is developed with Python3.6, Tensorflow 2.3 on a Ubuntu 18.04 OS

## Usage
Below is the command line for the train and test the CNN based attack method, when you run, you need to replace the pseudo parameters with your own path or specifications

Train:

python3 on-the-fly/train.py -i $data_path -o $output_dir -e $train_epoch_number -ns $samples_per_class -tb $traget_byte -aw $attack_window -tn $total_taining_number

Test:

python3 on-the-fly/test.py -i $data_path -m $model_root_dir -o $output_dir -tb $target_byte -aw $attack_window -tn $total_testing_number

## Contacts
Chenggang Wang: wang2c9@mail.uc.edu, University of Cincinnati

Boyang Wang: boyang.wang@uc.edu, University of Cincinnati
