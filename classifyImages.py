import os
# Parent Directory path
import pandas as pd
from glob import glob
from shutil import copyfile, copy
from sklearn.model_selection import train_test_split
import re


"""
Train will suppose the 60% of the data 
Validation will supppose 30% of the data
Test will only be 10% of all the data
"""
#Csv with values for the dataset
xray_data = pd.read_csv('/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/Data_Entry_2017.csv')

regexList =['.*Atelectasis.*', '.*Consolidation.*', '.*Infiltration.*', '.*Pneumothorax.*', '.*Edema.*', '.*Emphysema.*', '.*Fibrosis.*',
                    '.*Effusion.*', '.*Pneumonia.*', '.*Pleural_Thickening.*',
                    '.*Cardiomegaly.*', '.*Nodule.*', '.*Mass.*', '.*Hernia.*', '.*No Finding.*']
labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']
#Arrays for images paths
Atelectasis=[]
Consolidation=[]
Infiltration=[]
Pneumothorax=[]
Edema=[]
Emphysema=[]
Fibrosis=[]
Effusion=[]
Pneumonia=[]
Pleural_Thickening=[]
Cardiomegaly=[]
Nodule=[]
Mass=[]
Hernia=[]
No_Finding=[]

"""
Adds the path to the file to the data frame as a column called FullPath
"""
def add_full_path():
    my_glob = glob('/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/images*/images/*.png')
    full_img_paths = {os.path.basename(x): x for x in my_glob}
    #print(full_img_paths)
    xray_data['FullPath'] = xray_data['Image Index'].map(full_img_paths.get)

add_full_path()


"""
Adds a vector of 0 or 1 values corresponding to the label if it's present or not 
"""
def add_target_vector():
    dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']
    for label in dummy_labels:
        xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)

    xray_data['TargetVector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(
        lambda target: target[0])

add_target_vector()


train, testValidation = train_test_split(xray_data, test_size=0.40, random_state=100)
validation, test = train_test_split(testValidation, test_size=0.20, random_state=100)

# train, validate, test = pd.np.split(xray_data, [int(.6 * (xray_data)), int(.8 * len(xray_data))])
print(len(train), len(validation), len(test))
"""
Save Xray train test and validation dataframes as csv
"""
xray_data.to_csv('classifier/csvFiles/xrayCsv.csv', sep=",", index=False)
train.to_csv('classifier/csvFiles/trainCsv.csv', sep=",", index=False)
test.to_csv('classifier/csvFiles/testCsv.csv', sep=",", index=False)
validation.to_csv('classifier/csvFiles/validationCsv.csv', sep=",", index=False)



def create_labelled_dir():
    parent_dir_test = "/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/test"
    dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']
    for i in dummy_labels:
        if not os.path.exists(os.path.join(parent_dir_test,i)):
            os.makedirs(os.path.join(parent_dir_test,i))

    parent_dir_train = "/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/train"
    dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']
    for i in dummy_labels:
        if not os.path.exists(os.path.join(parent_dir_train,i)):
            os.makedirs(os.path.join(parent_dir_train,i))

    parent_dir_train = "/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/validation"
    dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']
    for i in dummy_labels:
        if not os.path.exists(os.path.join(parent_dir_train,i)):
            os.makedirs(os.path.join(parent_dir_train,i))

create_labelled_dir()



def fillDirectoryLabel(regex, label):
    """
    Si la finding label es Atelectasis entonces mover ese archivo desde su full path al nuevo path
    """
    labelsTest = test['Finding Labels'].tolist()
    labelsValidation = validation['Finding Labels'].tolist()
    labelsTrain = train['Finding Labels'].tolist()

    # print(labels)
    imagePathTest = test['FullPath'].tolist()
    imagePathValidation = validation['FullPath'].tolist()
    imagePathTrain = train['FullPath'].tolist()
    listToAddTest = []
    listToAddValidation = []
    listToAddTrain = []

    for ltest in range(len(labelsTest)):
        print(labelsTest[ltest])
        if re.search(regex,labelsTest[ltest]):  # regex
            listToAddTest.append(imagePathTest[ltest])

    for lval in range(len(labelsValidation)):
        print(labelsValidation[lval])
        if re.search(regex,labelsValidation[lval]):  # regex
            listToAddValidation.append(imagePathValidation[lval])

    for ltrain in range(len(labelsTrain)):
        print(labelsTrain[ltrain])
        if re.search(regex,labelsTrain[ltrain]):  # regex
            listToAddTrain.append(imagePathTrain[ltrain])

    print(len(listToAddTest), len(listToAddValidation), len(listToAddTrain))

    for t in range(len(listToAddTest)):
        print(listToAddTest[t])
        copy(listToAddTest[t], os.path.join('/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/test',label)) # Directory+label

    for v in range(len(listToAddValidation)):
        print(listToAddValidation[v])
        copy(listToAddValidation[v], os.path.join('/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/validation',label))  # Directory+label

    for tr in range(len(listToAddTrain)):
        print(listToAddTrain[tr])
        copy(listToAddTrain[tr], os.path.join('/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages/train',label))  # Directory+label

for regex in range(len(regexList)):
    print(regex)
    fillDirectoryLabel(regexList[regex],labels[regex])
print(xray_data)