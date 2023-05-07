# DTSNet-O2DTD
This repository provides the official implementation of the paper titled <b>“O2DTD: An annotated dataset for training deep learning systems for off-road open desert trail detection”</b> The paper has been submitted for consideration in the Nature Scientific Data journal.

## Introduction
A residual-learning-based asymmetric encoder-decoder network (RASP-Net) is proposed in this research. RASP-Net provides semantic segmentation and quantification of the following 11 OCT imaging-based chorioretinal biomarkers (CRBMs): (i) health (H), (ii) intraretinal fluid (IRF), (iii) subretinal fluid (SRF), (iv) serous pigment epithelial detachment (PED), (v) drusen/ reticular pseudodrusen (RPD), (vi) hard exudates or hyperreflective foci (HF), (vii) chorioretinal or geographic atrophy (GA), (viii) focal choroidal excavation (FCE), (ix) vitreomacular traction (VMT), (x) epiretinal membrane (ERM), and (xi) choroidal neovascular membrane (CNVM). RASP-Net operates at OCT B-scan level and requires pixel-wise annotations of 11 CRBMs against each scan. The overview of the proposed RASP-Net framework is presented below: 

<p align="center">
<img width=800 align="center" src = "https://github.com/BilalHassan90/OCT-Biomarker-Segmentation/blob/main/Images/Overview.jpg" alt="Introduction"> </br>
</p>

**Figure:** Overview of the proposed method. The RASP-Net framework integrated with coherent pre- and post-processing to perform the joint segmentation, quantification, and 3-D visualization of OCT imaging-based chorioretinal biomarkers.


This repository contains the source code of our propsed benhmark method DTSNet paper published in IEEE Transactions on Biomedical Engineering. The proposed framework is developed using <b>TensorFlow 2.3.1</b> and <b>Keras APIs</b> with <b>Python 3.7.8</b>. Moreover, the results are compiled through <b>MATLAB R2020a</b>. The detailed steps for installing and running the code are presented below:

## Installation
To run the codebase, following libraries are required. Although, the framework is developed using Anaconda. But it should be compatable with other platforms.

1) tensorflow 2.1.0
2) keras-gpu 2.3.1 
3) opencv 4.5.0
4) scipy 1.5.2
5) tqdm 4.65.0
6) imgaug 0.4.0 
7) matplotlib 3.4.3, and more

Alternatively, we also provide a yml file that contains all dependencies used for running the codes.

## Dataset
We have curated our own local dataset called Off-Road Open Desert Trail Detection (O2DTD), which is the first and largest dataset to focus on desert free space detection. We believe O2DTD dataset will help advance off-road autonomous driving and navigation systems. The dataset can be downloaded from the following link:

[O2DTD dataset](https://drive.google.com/file/d/1A-R5un-S6QiFb4nLzGhCzGB7hdqdrF0-/view?usp=sharing)

## Steps 
<p align="justify">
<b>Training the Model</b>
<p align="justify">
1) Create an environment using yml file and install any other packages if neccessary.
<p align="justify">
2) Download the O2DTD dataset from the link above.
<p align="justify">
3) Copy the training images and single-channel labels from the downloaded dataset in '…Data/TrainDataset/Images' and '…Data/TrainDataset/Labels/' folders, respectively.
<p align="justify">
4) Copy the validation images and single-channel labels from the downloaded dataset in '…Data/ValidationDataset/Images/' and '…Data/ValidationDataset/Labels/' folders, respectively.
<p align="justify">
5) Copy the test images and single-channel labels from the downloaded dataset in '…Data/TestDataset/Images/' and '…Data/TestDataset/Labels/' folders, respectively.
<p align="justify">
6) Open Trainer.py file, specify the learning rate (line 27), and optimizer (line 28 to line 30). For benchmarking, we have tried three different learning rates (0.1, 0.01, 0.001) and three different optimizers (SGD, Adadelta, Adam). By default, the Trainer.py file contains the hyperparameters that produced the best results.
<p align="justify">
7) Open Train.py file, choose the loss function (line 103 to 105), and update it in the line 107. For benchmarking, we have tried three different loss functions (cross-entropy, Dice, Tversky). By default, the Train.py file loads the Dice loss that produced best segmentation results.
<p align="justify">
8) Check all other parameters and directories and update them if required.
<p align="justify">
9) Run Trainer.py script to begin the training. 
<p align="justify">
10) After the training completes, model instance and training graph will be saved in the 'TrainedInstances' and 'Training Graph' folders, respectively. The segmented results on the test dataset will be stored in the '…Data/TestDataset/segmentation_results/' folder, and the results summary will be stored in the '…Data/TestDataset/results_summary/' folder.
<p align="justify">
11) We have also provided the best-segmentation results achieved on the test dataset, which can be downloaded from the following links:
</p>

[Single Channel](https://drive.google.com/file/d/1EYNhL9IvpVB2OhiWZ6bCtsHBsmP7cdQr/view?usp=sharing) 
[RGB](https://drive.google.com/file/d/1NqfeLfZdfSZBtKzP1HJYFgquuejcFI4f/view?usp=sharing)


<b>Using Trained Instance of the Model</b>
<p align="justify">
1) Repeat the steps 1 to 5, as mentioned above.
<p align="justify">
2) Open Evaluation.py file, load the trained model (line 183), and choose the loss function used during the model training (line 185 to 187).
<p align="justify">
3) Run the Evaluation.py script to segment the test images using saved model instance. 
<p align="justify">
4) After the evaluation completes, the segmented results along with the MAT files containing confidence scores will be stored in the '…Data/TestDataset/ResultsMATFiles//' folder. The results summary will be stored in the '…Data/TestDataset/results_summary/' folder. 

## Results
We have provided the quantitative andresults in the 'results' folder. Please contact us if you want to get the trained model instances.

## Citation
If you use RAG-Net<sub>v2</sub> (or any part of this code in your research), please cite the following paper:

```
@article{ragnetv2,
  title   = {Clinically Verified Hybrid Deep Learning System for Retinal Ganglion Cells Aware Grading of Glaucomatous Progression},
  author  = {Hina Raja and Taimur Hassan and Muhammad Usman Akram and Naoufel Werghi},
  journal = {IEEE Transactions on Biomedical Engineering},
  year = {2020}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae.
