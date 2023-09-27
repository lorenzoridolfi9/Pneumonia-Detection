# Pneumonia-Detection 🩻
## This repository contains medicla Image Classification project. Specifically, a classification model was created using a convolutional neural network (CNN), which can recognize from x-ray images, those who have pneumonia and those who are healthy.

## Dataset 📊
The dataset contains images of chests with pneumonia and healthy chests, is organized into 3 folders (train, test, val) and contains subfolders for each image category. There are 5,863 X-Ray images (JPEG) and 2 classes:
- Pneumonia
- Normal
  
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

you can find the dataset used in the project in my google drive at the following [Link](https://drive.google.com/drive/folders/1NS4rtssRgg5EGE6Mb0RiW87VylaBO4QK).

## Process of Analysis ⚙️
After loading the dataset, an exploratory data analysis was conducted to print on the screen some sample images present in the training set.
Then, since the classes were unbalanced, to avoid overfitting problems, a data augmentation technique was implemented through the ImageDataAugmentation library, which allows replicating the images present in the dataset by modifying some of their details:
- causal rotation of the images
- moving the images vertically and horizontally

Replicas of images whose class was numerically lower were created.
