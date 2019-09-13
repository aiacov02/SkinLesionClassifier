
import numpy as np
import itertools


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import confusion_matrix, recall_score, auc, roc_auc_score, roc_curve, mean_squared_error, balanced_accuracy_score, precision_score, f1_score




import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# encode the age into integer
def encode_age(age):
    if (age > 0 and age < 40):
        return 0
    elif (age >= 40 and age <65):
        return 1
    else:
        return 2


# encode the category into integer
def encode_category(cat):
    if cat == "akiec":
        return 0
    elif cat == "bcc":
        return 1
    elif cat == "bkl":
        return 2
    elif cat == "df":
        return 3
    elif cat == "mel":
        return 4
    elif cat == "nv":
        return 5
    elif cat == "vasc":
        return 6

# encode the sex into integer
def encode_sex(sex):
    if sex == "male":
        return 0
    else:
        return 1

# Load the training data and prepare them
def load_data():
    df_densenet = pd.read_csv('prediction_densenet121_copy.csv')


    cols = df_densenet.columns.tolist()

    cols = cols[-1:] + cols[:-1]

    df_densenet = df_densenet[cols]


    df_inception = pd.read_csv('prediction_inceptionv3_copy.csv')
    df_resnet = pd.read_csv('prediction_resnet50_copy.csv')
    df_vgg16 = pd.read_csv('prediction_vgg16_copy.csv')

    df_metadata = pd.read_csv('train.csv')

    df_metadata = df_metadata.drop(['lesion_id', 'dx_type'], axis=1)

    df_metadata.rename(columns={'image_id':'Filename'}, inplace=True)

    df_preds = pd.merge(df_densenet,df_inception, on="Filename",how="inner")
    df_preds = pd.merge(df_preds,df_resnet, on="Filename",how="inner")
    df_preds = pd.merge(df_preds,df_vgg16, on="Filename",how="inner")

    df_preds['Filename'] = df_preds['Filename'].map(lambda line: line.split('/', 1)[-1]).map(lambda line: line.split('.',1)[0])

    df_preds = pd.merge(df_preds,df_metadata, on="Filename",how="inner")

    df_preds['age'].replace('', np.nan, inplace=True)
    df_preds['age'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['age'], inplace=True)

    df_preds['age'] = df_preds['age'].map(lambda age: encode_age(age))

    df_preds['DenseNet121_Predictions'] = df_preds['DenseNet121_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['InceptionV3_Predictions'] = df_preds['InceptionV3_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['ResNet50_Predictions'] = df_preds['ResNet50_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['Vgg_Predictions'] = df_preds['Vgg_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['dx'] = df_preds['dx'].map(lambda cat: encode_category(cat))


    df_preds['sex'].replace('', np.nan, inplace=True)
    df_preds['sex'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['sex'], inplace=True)

    df_preds['sex'] = df_preds['sex'].map(lambda sex: encode_sex(sex))

    df_preds['localization'].replace('', np.nan, inplace=True)
    df_preds['localization'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['localization'], inplace=True)

    df_preds['localization'] = LabelEncoder().fit_transform(df_preds['localization'])

    return df_preds

# Load and prepare the test data
def load_test_data():
    df_densenet = pd.read_csv('test_prediction_densenet121.csv')


    cols = df_densenet.columns.tolist()

    cols = cols[-1:] + cols[:-1]

    df_densenet = df_densenet[cols]


    df_inception = pd.read_csv('test_prediction_inceptionv3.csv')
    df_resnet = pd.read_csv('test_prediction_resnet50.csv')
    df_vgg16 = pd.read_csv('test_prediction_vgg16.csv')

    df_metadata = pd.read_csv('train.csv')

    df_metadata = df_metadata.drop(['lesion_id', 'dx_type'], axis=1)

    df_metadata.rename(columns={'image_id':'Filename'}, inplace=True)

    df_preds = pd.merge(df_densenet,df_inception, on="Filename",how="inner")
    df_preds = pd.merge(df_preds,df_resnet, on="Filename",how="inner")
    df_preds = pd.merge(df_preds,df_vgg16, on="Filename",how="inner")

    df_preds['Filename'] = df_preds['Filename'].map(lambda line: line.split('/', 1)[-1]).map(lambda line: line.split('.',1)[0])

    df_preds = pd.merge(df_preds,df_metadata, on="Filename",how="inner")

    df_preds['age'].replace('', np.nan, inplace=True)
    df_preds['age'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['age'], inplace=True)


    df_preds['age'] = df_preds['age'].map(lambda age: encode_age(age))

    df_preds['DenseNet121_Predictions'] = df_preds['DenseNet121_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['InceptionV3_Predictions'] = df_preds['InceptionV3_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['ResNet50_Predictions'] = df_preds['ResNet50_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['Vgg_Predictions'] = df_preds['Vgg_Predictions'].map(lambda cat: encode_category(cat))
    df_preds['dx'] = df_preds['dx'].map(lambda cat: encode_category(cat))


    df_preds['sex'].replace('', np.nan, inplace=True)
    df_preds['sex'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['sex'], inplace=True)

    df_preds['sex'] = df_preds['sex'].map(lambda sex: encode_sex(sex))

    df_preds['localization'].replace('', np.nan, inplace=True)
    df_preds['localization'].replace('unknown', np.nan, inplace=True)

    df_preds.dropna(subset=['localization'], inplace=True)

    df_preds['localization'] = LabelEncoder().fit_transform(df_preds['localization'])

    return df_preds


# train the xg_boost model and make predictions
def train_xgboost(df_preds, df_preds2):
    df_preds = df_preds.drop(['Filename'], axis=1)

    df_preds = df_preds[['DenseNet121_Predictions','InceptionV3_Predictions','ResNet50_Predictions','Vgg_Predictions','sex', 'localization', 'age','dx']]


    X,y = df_preds.iloc[:,:-1],df_preds.iloc[:,-1]


    print(X.head())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=None, random_state=50 ,shuffle=True)


    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)



    params = {"objective":"multi:softmax", "num_class":7 ,'colsample_bytree': 0.3,'learning_rate': 0.001,
              'max_depth': 100, 'alpha': 10, "n_estimators":100}


    xg_class = xgb.XGBClassifier(objective ='mult:softmax', num_class=7, colsample_bytree = 0.03, learning_rate = 0.1,
                              max_depth = 1000, alpha = 1000, n_estimators = 1000)

    xg_class.fit(X_train,y_train)


    df_preds2 = df_preds2.drop(['Filename'], axis=1)

    df_preds2 = df_preds2[['DenseNet121_Predictions','InceptionV3_Predictions','ResNet50_Predictions','Vgg_Predictions','sex', 'localization', 'age','dx']]


    X,y = df_preds2.iloc[:,:-1],df_preds2.iloc[:,-1]



    print(X.head())

    data_dmatrix = xgb.DMatrix(data=X,label=y)

    X_test = X
    y_test = y

    preds = xg_class.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

    plot_confusion_matrix(cm, cm_plot_labels, "xgboost")

    print("Balanced Accuracy: " + str(balanced_accuracy_score(y_test, preds)))
    print("Weighted Recall: " + str(recall_score(y_test, preds, average='weighted')))
    print("Class Recall: " + str(recall_score(y_test, preds, average=None)))
    print("Weighted Precision: " + str(precision_score(y_test, preds, average='weighted')))
    print("Class Precision: " + str(precision_score(y_test, preds, average=None)))
    print("Mean f1 score " + str(f1_score(y_test, preds, average='weighted')))
    print("Class f1 score " + str(f1_score(y_test, preds, average=None)))


    file = open("xgboost_results_Original.txt","w+")

    file.write("Balanced Accuracy: " + str(balanced_accuracy_score(y_test, preds)) + "\n")
    file.write("Weighted Recall: " + str(recall_score(y_test, preds, average='weighted')) + "\n")
    file.write("Class Recall: " + str(recall_score(y_test, preds, average=None)) + "\n")
    file.write("Weighted Precision: " + str(precision_score(y_test, preds, average='weighted')) + "\n")
    file.write("Class Precision: " + str(precision_score(y_test, preds, average=None)) + "\n")
    file.write("Mean f1 score " + str(f1_score(y_test, preds, average='weighted')))
    file.write("Class f1 score " + str(f1_score(y_test, preds, average=None)))


    file.close()

    xg_class = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)


    ax = xgb.plot_importance(xg_class)
    fig = ax.figure
    fig.set_size_inches(20,20)
    # plt.savefig('importance.png')



# Declare a function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, classifier, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('XGBooster_confusion_matrix_original.png')



if __name__ == '__main__':
    df_preds = load_data()
    df_preds2 = load_test_data()
    train_xgboost(df_preds,df_preds2)

