# Skin Lesion Image Classification Using Convolution Neural Networks and Patient Meta-Data

This is a tool for the automatic classification of skin lesion images, using images and the patient's meta data.


The tool described in this project, aims at helping in the automatic classification of skin lesion images, by making the following contributions: 

1. Assess the best-performing transfer-learning plan for Convolutional Neural Networks, for the classification of skin images. A variety of different pretrained Convolutional Neural Networks were trained and fined tuned, using different techniques, such as adding or freezing layers etc. The performance of the models was compared to assess the best-performing one. The use of machine learning was also proposed, to combine the predicting power of all the Convolution Neural Networks.  

2. Propose the use of patient meta-data to improve classification performance. Instead of just using image classification techniques for the classification of skin lesions, it is proposed that patient meta-data can also be used, in a way to improve the performance. The developed tool uses machine learning techniques to combine predictions of the image classification models as well as the patient meta-data.


Dataset:

Our data were extracted from the “ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection” grand challenge datasets (Codella, et al., 2017) (Tschandl, et al., 2018). 

The dataset consists of JPEG images belonging to 7 different skin lesion categories or classes: 

1. Melanoma (mel) 

2. Melanocytic nevus (nv) 

3. Basal cell carcinoma (bcc) 

4. Actinic keratosis / Bowen’s disease (akiec) 

5. Benign keratosis (bkl) 

6. Dermatofibroma (df) 

7. Vascular lesion (vasc)

<br />

Architecture:

![Architectural Diagram](/architecture.png)


<br />

Instructions to use:
1. Make sure you have an NVIDIA GPU with enough memory, that supports cuda and has cuda installed
2. Download the training dataset from the ISIC 2018 Skin Lesion Image Classification Challenge
3. Place images in directory data or change the directory in the preprocessing.py file
4. Place the csv ground truth file called train.csv in the root directory of the project
5. Run "pip install -r requirements.txt" to install the requirements to run the program
6. Create directory start_dir
7. Run "python preprocessing.py" to perform image pre-processing
8. Run "python trainCNNs.py" to train the 4 CNN models
9. Run "predictTraining.py" to predict on the ensemble training data
10. Run "predict.py" to predict on the ensemble test data
11. Run "trainEnsemble.py" to train and predict on the test data using the ensemble
12. Run "trainEnsembleCompare.py" to compare the ensemble results with and without meta-data
