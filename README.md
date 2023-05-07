# DTSNet-O2DTD
This repository provides the official implementation of the paper titled <b>“O2DTD: An annotated dataset for training deep learning systems for off-road open desert trail detection”</b> The paper has been submitted for consideration in the Nature Scientific Data journal.

## Citation
If you use any part of the provided code in your research, please consider citing the paper as follows:
```

}
```
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
<b>For Training the Model</b>

1) Create an environment using yml file and install any other packages if neccessary.
2) Download the O2DTD dataset from the link above.
3) Copy the training images and single-channel labels from the downloaded dataset in '…Data/TrainDataset/Images' and '…Data/TrainDataset/Labels/' folders, respectively.
4) Copy the validation images and single-channel labels from the downloaded dataset in '…Data/ValidationDataset/Images/' and '…Data/ValidationDataset/Labels/' folders, respectively.
5) Copy the test images and single-channel labels from the downloaded dataset in '…Data/TestDataset/Images/' and '…Data/TestDataset/Labels/' folders, respectively.

	Use 'augmentation.py' or 'augmentor.m' to augment the training scans
4) Put the augmented training images in '…\trainingDataset\train_images' and '…\codebase\models\trainingSet' folders. The former one is used for segmentation and the latter one is used for the classification purposes.
5) Put the training annotations (for segmentation) in '…\trainingDataset\train_annotations' folder
6) Put validation images in '…\trainingDataset\val_images' and '…\codebase\models\validationSet' folders. The former one is used for segmentation and the latter one is used for the classification purposes.
7) Put validation annotations (for segmentation) in '…\trainingDataset\val_annotations' folder. Note: the images and annotations should have same name and extension (preferably .png).
8) Put test images in '…\testingDataset\test_images' folder and their annotations in '…\testingDataset\test_annotations' folder
9) Use 'trainer.py' file to train RAG-Net<sub>v2</sub> on preprocessed scans and also to evaluate the trained model on test scans. The results on the test scans are saved in ‘…\testingDataset\segmentation_results’ folder. This script also saves the trained model in 'model.h5' file.
10) Run 'ragClassifier.py' script to classify the preprocessed test scans as normal or glaucomic. The results are saved as a mat file in '…\testingDataset\diagnosisResults' folder. Note: step 10 can only be done once the step 9 is finished because the model trained in step 9 is required in step 10. 
11) Once step 10 is completed, run 'trainSVM.m' script to train the SVM model for grading the severity of the classified glaucomic scans.
12) Once the SVM is trained, run 'glaucomaGrader.m' to get the grading results.
13) The trained models can also be ported to MATLAB using ‘kerasConverter.m’ (this step is optional and only designed to facilitate MATLAB users if they want to avoid Python analysis).
14) Some additional results (both qualitative and quantitative) of the proposed framework are also presented in the '…\results' folder. 

<b>For Fundus Analysis</b>

15) Download the desired dataset
16) Put the training scans in '…\codebase\models\trainingSet\glaucoma' and '…\codebase\models\trainingSet\normal' folders
17) Put the validation scans in '…\codebase\models\validationSet\glaucoma' and '…\codebase\models\validationSet\normal' folders
18) Put the test scans in '…\testingDataset\fundus_test_images' folder.
19) Uncomment the path at line 44 within the 'ragClassifier.py' file. Note: if you want to perform OCT analysis again, this line has to be commented again
20) Run 'ragClassifier.py' to produce classify normal and glaucomic fundus scans. The results will saved in a mat file within  '…\testingDataset\diagnosisResults' folder once the analysis is completed. Note: Before running 'ragClassifier.py', please make sure you have the saved 'model.h5' file generated through step 9 because RAG-Net<sub>v2</sub> classification model initially adapts the weights of the trained RAG-Net<sub>v2</sub> segmentation unit for faster convergence.
</p>

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


## Prerequisites
MATLAB R2020a platform with deep learning, image processing, and computer vision toolboxes. 

## Stepwise Operations
We provide separate main files for four operations, including preprocessing, network training and validation, postprocessing, and quantification.

<p align="justify">
<b>Data Preprocessing </b>

  1.	Put the raw OCT scans data in the “…\Raw Scans” folder and pixel-wise ground truth annotations in the “…\Ground Truth Labels” folder. The label IDs corresponding to each class pixel are provided in the “Classes_ID.mat” file.
  2.	To preprocess the scans, use the “Preprocessor.m” file. The scans containing VMT CRBM are preprocessed differently. Please select the option “Yes” if the candidate OCT scan has the VMT CRBM and “No” otherwise. The preprocessed scans are stored in the “…\Preprocessed” folder. The values of preprocessing parameters are empirically adjusted, generating adequate results in most cases. 

<b>Network Training and Validation </b>

  3.	The network requires the preprocessed scans for training as stored in the “…\Preprocessed” folder in the previous step.
  4.	To train the network from scratch, use the “Trainingcode.m” file and specify the training hyper-parameters. The data is split in the ratio of 60:20:20 for the train, validate, and test subsets. The IDs of each relevant subset are stored in the “Idx.mat” file. 
  5.	Once the network training is completed, the trained instances are saved as a “TrainedNet.mat” file. While the predicted labels are stored in the “…\Predicted Labels” folder.
  
<b>Data Postprocessing </b>

  6.  In the next step, the network predicted results are cleaned using the postprocessing scheme. For this purpose, use the “Postprocessing.m” file.
  7.  This step requires the predicted scans for postprocessing stored in the “…\Predicted Labels” folder in the previous step.
  8.  The final postprocessed scans are stored in the “…\PostProcessed” folder. 

<b>CRBMs Quantification </b>

  9.  The quantification of CRBMs can be performed at the B-scan level or the eye level using OCT volumes.
  10. This step requires the postprocessed scans stored in the “...\OCT Volumes\1\Postprocessed” folder, generated using the postprocessing scheme. Put the corresponding ground truth labels in the “...\OCT Volumes\1\Ground Truth Labels” folder.
  11. Run the “Quantification.m” file for CRBMs quantification. This step also generates the 3D macular profile of the candidate OCT volume along with the quantification results and saves them in the “...\OCT Volumes\1\3DQuantification” folder.
</p>

## Results
We have provided the results of 20 sample OCT scans in the “...\Other” directory.

## Contact
If you have any query, please feel free to contact us at bilalhassan@buaa.edu.cn 

	



  
