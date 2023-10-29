import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import backend as k
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, auc, roc_curve
from PIL import Image

import matplotlib
matplotlib.use('Agg') # this is for saving the images using a non-interactive backend Agg
import matplotlib.pyplot as plt


# Script for first experiment. 10-Fold divided images
 
modelName = ['Xception', 'InceptionV3', 'VGG16', 'VGG19', 'ResNet50']
batch_size = 8  

 
 
def classify(trained_model_dir, test_data, label_test):
 
    # load json and create model
    json_file = open(os.path.join(trained_model_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(trained_model_dir, model_weights))

    print('Number of test images: ' + str(len(test_data)))

    test_data = [im / 255 for im in test_data]
    test_data = np.array(test_data).astype(np.float64)
    y_probabilities = model.predict(test_data)

    probs_test = []
    for i, prob in enumerate(y_probabilities):
        if prob[0] > 0.5:
            probs_test.append(1)
        else:
            probs_test.append(0)

    # print(probs_test)
    # score = accuracy_score(label_test, probs_test)
    fpr, tpr, _ = roc_curve(label_test, y_probabilities[:,0])
    auc_score = auc(fpr, tpr)

    return auc_score
 


def load_images(data_dir, img_width, img_height):

    allNames = []
    allImages = []
    allLabels = []

    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith((".jpg", ".png", "JPG")):
                # filenames
                allNames.append(name)
                # Reading images
                im = np.asarray(Image.open(root + '/' + name).resize((img_width, img_height))) # We read and resize the image
                allImages.append(im)
                # Creating labels from names
                if '_g_' in name:
                    allLabels.append(1)
                else:
                    allLabels.append(0)

    return allImages, allLabels, allNames


if __name__ == '__main__':

    for m in modelName:

        print('\n ---------------------     Doing bootstrap for ' + m)
        if 'ception' in m:
            img_width, img_height = 299, 299  # 224x224 for VGG16, VGG19, and ResNet50. 299x299 for InceptionV3 and Xception
        else:
            img_width, img_height = 224, 224  

        lowerVals = []
        higherVals = []

        for fold in range(0,10):

            print('\n ---------------------     Doing for Fold: ' + str(fold))

            model_dir = 'ModelsByFold/' + m + '/' + 'Fold' + str(fold) + '/' # each class should have it's own folder
            results_dir = 'results/'

            # default paths
            model_name = 'Fold_' + str(fold) + 'model.json'
            model_weights = 'Fold_' + str(fold) + m + '_Batch8final_weights.h5'

            per_train = 0.90 # percentage for test set

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            data_dir = '../imagesInFolds_OnlyPublic/Fold' + str(fold) + '/test/'
            images, labels, allNames = load_images(data_dir, img_width, img_height)

            # configure bootstrap
            n_iterations = 10
            n_size = int(len(images) * per_train)
            # run bootstrap
            stats = list()
            for i in range(n_iterations):

                print('\n     -----   Running for iteration ' + str(i+1) + ' of ' + str(n_iterations))

                # prepare test set
                x_test, fnames_test, y_test = resample(images, allNames, labels, replace=False, n_samples=n_size)

                print('Size test data:' + str(len(x_test)))
                print('Size test labels:' + str(len(y_test)))
                print('Size filenames:' + str(len(fnames_test)))

                # evaluate model
                score = classify(model_dir, x_test, y_test)  # trained model
                score = float('%.4g' % score) # Take only 4 significant digits

                print('Accuracy value for test images:' + str(score))
                stats.append(score)

                k.clear_session()

            # confidence intervals
            alpha = 0.95
            p = ((1.0-alpha)/2.0) * 100
            lower = max(0.0, np.percentile(stats, p))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print('%.1f confidence interval %.2f%% and %.2f%%' % (alpha*100, lower*100, upper*100))

            lowerVals.append(lower)
            higherVals.append(upper)

        print('Average for lower values: Fold: ' + str(fold) + ' -> ' + str( float('%.4g' % np.mean(lowerVals))) )
        print('Average for higher values: Fold: ' + str(fold) + ' -> ' + str( float('%.4g' % np.mean(higherVals))))
        print('%.1f Average confidence interval %.2f%% and %.2f%%' % (0.95 * 100, float('%.4g' % np.mean(lowerVals)) * 100, float('%.4g' % np.mean(higherVals)) * 100))