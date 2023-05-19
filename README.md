# DTSNet-O2DTD
This repository provides the official implementation of the paper titled <b>“O2DTD: An annotated dataset for training deep learning systems for off-road open desert trail detection”</b>. The paper has been submitted for consideration in the Nature Scientific Data journal.

## Introduction
In this work, we address the literature gap related to free space detection in off-road scenarios, particularly in desert environments, and present a new dataset and deep learning model that can be used as a benchmark for future research in this area. Overall, this work represents a significant step towards enhancing the capabilities of autonomous driving and navigation systems in off-road scenarios, where free space detection is critical for safe and efficient operation.


<p align="center">
<img width=800 align="center" src = "https://github.com/BilalHassan90/DTSNet-O2DTD/blob/main/Other/Fig.png" alt="Introduction"> </br>
</p>

**Figure:** Overview of the proposed methodology for O2DTD dataset curation and benchmarking. (a) Dataset collection, (b) Dataset annotation, (c) Dataset validation, (d) Upload to the data repository.


This repository contains the source code of our proposed benchmark method DTSNet, submitted for consideration in the Nature Scientific Data journal. The proposed framework is developed using <b>TensorFlow 2.3.1</b> and <b>Keras APIs</b> with <b>Python 3.7.8</b>. The detailed steps for installing and running the code are presented below:

## Installation
The following libraries are required to run the codebase. Although the framework is developed using Anaconda. But it should be compatible with other platforms.

1) tensorflow 2.1.0
2) keras-gpu 2.3.1 
3) opencv 4.6.0
4) tqdm 4.65.0
5) matplotlib 3.4.3, and more

Alternatively, we also provide a yml file that contains all dependencies used for running the codes.

## Dataset
We have curated our local dataset called Off-Road Open Desert Trail Detection (O2DTD), the first and largest dataset to focus on desert free space detection. We believe the O2DTD dataset will help advance off-road autonomous driving and navigation systems. The dataset can be downloaded from the following link:

[O2DTD dataset](https://drive.google.com/file/d/1A-R5un-S6QiFb4nLzGhCzGB7hdqdrF0-/view?usp=share_link)

## Steps 
<p align="justify">
<b>Training the Model</b>
<p align="justify">
1) Create an environment using the yml file and install any other packages if necessary.
<p align="justify">
2) Download the O2DTD dataset from the link above.
<p align="justify">
3) Copy the training images and single-channel labels from the downloaded dataset in '…Data/TrainDataset/Images' and '…Data/TrainDataset/Labels/' folders, respectively.
<p align="justify">
4) Copy the validation images and single-channel labels from the downloaded dataset in '…Data/ValidationDataset/Images/' and '…Data/ValidationDataset/Labels/' folders, respectively.
<p align="justify">
5) Copy the test images and single-channel labels from the downloaded dataset in '…Data/TestDataset/Images/' and '…Data/TestDataset/Labels/' folders, respectively.
<p align="justify">
6) Open the Trainer.py file, specify the learning rate (line 27) and optimizer (line 28 to line 30). We have tried three different learning rates (0.1, 0.01, 0.001) and three different optimizers (SGD, Adadelta, Adam) for benchmarking. By default, the Trainer.py file contains the hyperparameters that produced the best results.
<p align="justify">
7) Open the Train.py file, choose the loss function (lines 103 to 105), and update it in line 107. For benchmarking, we have tried three different loss functions (cross-entropy, Dice, and Tversky). By default, the Train.py file loads the Dice loss that produced the best segmentation results.
<p align="justify">
8) Check all other parameters and directories and update them if required.
<p align="justify">
9) Run the Trainer.py script to begin the training. 
<p align="justify">
10) After the training completes, the model instance and training graph will be saved in the 'TrainedInstances' and 'Training Graph' folders, respectively. The segmented results on the test dataset will be stored in the '…Data/TestDataset/segmentation_results/' folder and the results summary will be stored in the '…Data/TestDataset/results_summary/' folder.


<b>Using Trained Instance of the Model</b>
<p align="justify">
1) Repeat the steps 1 to 5, as mentioned above. </p>
<p align="justify">
2) Download the trained instances from the below link, and put them in the 'TrainedInstance' folder.</p>

[Trained Instances](https://drive.google.com/drive/folders/1k5-xei0G9GUs0eRLErF90uwmlKdwWxhD)

<p align="justify">
3) Open the Evaluation.py file, load the trained model (line 183), and choose the loss function used during the model training (lines 185 to 187).
<p align="justify">
4) Run the Evaluation.py script to segment the test images using the saved model instance. 
<p align="justify">
5) After the evaluation completes, the segmented results along with the MAT files containing confidence scores, will be stored in the '…Data/TestDataset/ResultsMATFiles//' folder. The results summary will be stored in the '…Data/TestDataset/results_summary/' folder. 

## Results
<p align="justify">
We have also provided the best-segmentation results achieved on the test dataset, which can be downloaded from the following links:
</p>

[Single Channel](https://drive.google.com/file/d/1EYNhL9IvpVB2OhiWZ6bCtsHBsmP7cdQr/view?usp=sharing), 
[RGB](https://drive.google.com/file/d/1NqfeLfZdfSZBtKzP1HJYFgquuejcFI4f/view?usp=sharing)

## Citation
If you use the O2DTD dataset, DTSNet model, or any part of this code in your research, please cite the following paper:

```
@article{O2DTD2023,
  title   = {O2DTD: An annotated dataset for training deep learning systems for off-road open desert trail detection},
  author  = {Hassan, Bilal and AlRemeithi, Hamad and Ahmed, Ramsha and Zayer, Fakhreddine and Hassan, Taimur and Khonji, Majid Khonji and Dias, Jorge},
  journal = {Scientific Data},
  year = {2023},
  publisher={Nature}
}
```

## Contact
If you have any queries, please contact us at: bilal.hassan@ku.ac.ae.
