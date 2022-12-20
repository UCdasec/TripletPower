# TripletPower

** The dataset and code are for research purpose only**

TripletPower can reduce the number of training traces for deep-learning side-channel attacks. Specifically, TripletPower leverages Triplet Networks (a type of deep metric learning) and learn robust feature space with a fewer number of traces. As a result, TripletPower only needs hundreds of training traces to train a ML model to recover AES encryption keys on microcontrollers while a CNN requires thousands of traces. We observe promising results from AVR XMEGA (8-bit) and also ARM STM32 (32-bit) microcontrollers. In addition to profiling attacks in both same-device and cross-device settings, TripletPower can also be extended to non-profiling attacks to succesfully recover AES encryption keys with on-the-fly labeling. More details can be found in our HOST'23 paper below. 

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Chenggang Wang, Jimmy Dani, Shane Reilly, Austhen Brownfield, Boyang Wang, John Emmert, "TripletPower: Deep-Learning Side-Channel Attacks over Few Traces," IEEE International Symposium on Hardware Oriented Security and Trust (HOST 2023), San Jose, CA, USA, May 1-4 2023

## Code
The codebased include 3 folders: cnn, triplet and tools
>
> - cnn: include the codes for runing cnn method
> - triplet: include the codes for runing triplet method
> - tools: include the codes of all support functions
>

## Datasets
Our datasets used in this study can be accessed through the link below (last modified Dec. 2022): 

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EtdJJGogSrBPvGOkq_d_YHIBd1eStajTUDnYZ-UD9LiA4w?e=BQechV

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!

## How to Reproduce the results
1. For CNN based method, please follow the description in cnn/README.md
2. For Triplet based method, please follow the description in triplet/README.md

# Contacts
Chenggang Wang wang2c9@mail.uc.edu

Boyang Wang wang2ba@ucmail.uc.edu

Austen Brownfield brownfaw@mail.uc.edu
