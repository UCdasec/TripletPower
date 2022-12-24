# Triplet Based Method
This folder contains the code for triplet based side-channel attack method. In this file, it explains what function that each source code file have and what envrionment that the code can run on, and also give detailed the command line of train and test the TripletPower side-channel attack model.

## Content
This instructions for each python source code file is list below:

test.py: This is the source code for testing TripletPower based side-channel attack model
triplet.py: This is the source code for training TripletPower based side-channel attack model


## Data
For the data download, please refer the README.md at the root folder

## Requirement
This project is developed with Python3.6, Tensorflow 2.3 on a Ubuntu 18.04 OS

## Usage
Below is the command line for the train and test the CNN based attack method, when you run, you need to replace the pseudo parameters with your own path or specifications

Train:

python3 triplet/triplet.py -i $data -o $outDir/125 -e $epoch_num -ns $samples_per_class -tb $target_byte -lm $leakage_model -aw $attack_window -tn $train_data_number

Test:

python3 triplet/test.py -i $data -rd $result_output_dir -tn $testing_data_number -tb $target_byte -lm $leakage_model -aw $attack_window

## Contacts
Chenggang Wang: wang2c9@mail.uc.edu, University of Cincinnati

Boyang Wang: boyang.wang@uc.edu, University of Cincinnati
