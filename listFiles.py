import os, os.path
import logging
logging.basicConfig(filename='logs/listFiles.log', filemode='w',format='%(asctime)s - %(message)s',level=logging.INFO)

genericPath = '/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages'
labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                    'Effusion', 'Pneumonia', 'Pleural_Thickening',
                    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No_Finding']

def get_train_images():
    for label in labels:
        path= os.path.join(genericPath,'train' ,label)
        logging.info('The number of files for Training on ' + label +' '+ str(len(
            [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])))
        print('The number of files for Training on '+label)
        print(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
def get_test_images():
    for label in labels:
        path = os.path.join(genericPath, 'test', label)
        logging.info('The number of files for Test on ' + label +' '+str(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])))
        print('The number of files for Test on ' + label)
        print(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
def get_validation_images():
    for label in labels:
        path = os.path.join(genericPath, 'validation', label)
        logging.info('The number of files for Validation on ' + label +' '+ str(len(
            [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])))
        print('The number of files for Validation on ' + label)
        print(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
def get_number_images_total_per_class():
    for label in labels:
        pathTrain = os.path.join(genericPath,'train' ,label)
        pathTest = os.path.join(genericPath, 'test', label)
        pathValidation = os.path.join(genericPath, 'validation', label)
        countTrain = len([name for name in os.listdir(pathTrain) if os.path.isfile(os.path.join(pathTrain, name))])
        countTest = len([name for name in os.listdir(pathTest) if os.path.isfile(os.path.join(pathTest, name))])
        countValidation = len([name for name in os.listdir(pathValidation) if os.path.isfile(os.path.join(pathValidation, name))])
        total = countTest + countTrain + countValidation
        logging.info('The number of files for'+label+' '+str(total))
if __name__ == "__main__":
    get_train_images()
    get_test_images()
    get_validation_images()
    get_number_images_total_per_class()