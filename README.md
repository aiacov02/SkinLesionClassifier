# Skin Lesion Image Classification

This is a tool for the automatic classification of skin lesion images, using images and the patient's meta data.

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