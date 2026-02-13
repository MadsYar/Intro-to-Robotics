#
# file  CS490_Assignment_1.py
# brief Purdue University Fall 2022 CS490 robotics Assignment 1 -
#       Gaussian Discriminant Analysis
# date  2022-09-01
#

#you can only import modules listed in the handout
import os
import sys
import math
from roipoly import RoiPoly
import numpy as np
import matplotlib.pyplot as plt

gda_type = None
#hand label region related functions
#**************************************************************************************************
#hand label pos/neg region for training data
#write regions to files, this function should not return anything
def label_training_dataset(training_path, region_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(base_dir, training_path)
    region_path   = os.path.join(base_dir, region_path)

    os.makedirs(region_path, exist_ok=True)

    for root, _, files in os.walk(training_path):
        for name in files:
            base = os.path.splitext(name)[0]

            positive_path = os.path.join(region_path, f"{base}_pos.npy")
            negative_path = os.path.join(region_path, f"{base}_neg.npy")

            img = plt.imread(os.path.join(root, name))

            # Barrel regions
            plt.figure()
            plt.imshow(img)
            plt.title("Picture: " + f"{base}")
            positive_roi = RoiPoly(color='r')

            # Non-barrel regions
            plt.clf()
            plt.imshow(img)
            positive_roi.display_roi() 
            plt.title("Picture: " + f"{base}")
            negative_roi = RoiPoly(color='b')

            positive_features = img[positive_roi.get_mask(img[:, :, 0] if img.ndim == 3 else img)]  # Extract RGB values
            negative_features = img[negative_roi.get_mask(img[:, :, 0] if img.ndim == 3 else img)]  # Extract RGB values
            np.save(positive_path, positive_features)
            np.save(negative_path, negative_features)

            plt.close()

            print(f"[saved] {os.path.basename(positive_path)}, {os.path.basename(negative_path)}")

    pass

#hand label pos region for testing data
#write regions to files, this function should not return anything
def label_testing_dataset(training_path, region_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(base_dir, training_path)
    region_path   = os.path.join(base_dir, region_path)

    os.makedirs(region_path, exist_ok=True)

    for root, _, files in os.walk(training_path):
        for name in files:

            base = os.path.splitext(name)[0]

            positive_path = os.path.join(region_path, f"{base}_pos.npy")
            negative_path = os.path.join(region_path, f"{base}_neg.npy")

            img = plt.imread(os.path.join(root, name))

            # Barrel regions
            plt.figure()
            plt.imshow(img)
            plt.title("Picture: " + f"{base}")
            positive_roi = RoiPoly(color='r')

            positive_features = img[positive_roi.get_mask(img[:, :, 0] if img.ndim == 3 else img)]  # Extract RGB values
            negative_features = img[~(positive_roi.get_mask(img[:, :, 0] if img.ndim == 3 else img))]  # Extract RGB values
            np.save(positive_path, positive_features)
            np.save(negative_path, negative_features)

            plt.close()

            print(f"[saved] {os.path.basename(positive_path)}, {os.path.basename(negative_path)}")
    pass
#**************************************************************************************************


#import labeled regions related functions
#**************************************************************************************************
#import pre hand labeled region for trainning data
def import_pre_labeled_training(training_path, region_path):
    features, labels = None, None
    features_all, labels_all = [], []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    region_path = os.path.join(base_dir, region_path)
    
    for filename in sorted(os.listdir(region_path)):
        feature_file = np.load(os.path.join(region_path, filename))
        labels_file = np.ones(len(feature_file)) if '_pos' in filename else np.zeros(len(feature_file))

        features_all.append(feature_file)
        labels_all.append(labels_file)

    if features_all:
        features = np.vstack(features_all)
        labels = np.concatenate(labels_all)
    
    return features, labels

#import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    features, labels = None, None
    features_all, labels_all = [], []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    region_path = os.path.join(base_dir, region_path)
    
    region_files = sorted(os.listdir(region_path))
    
    for filename in region_files:
        feature_file = np.load(os.path.join(region_path, filename))
        labels_file = np.ones(len(feature_file)) if '_pos' in filename else np.zeros(len(feature_file))
        
        features_all.append(feature_file)
        labels_all.append(labels_file)

    if features_all:
        features = np.vstack(features_all)
        labels = np.concatenate(labels_all)
    
    return features, labels
#**************************************************************************************************


#main GDA training functions
#**************************************************************************************************
def train_GDA_common_variance(features, labels):
    prior, mu, cov = None, None, None
    features_0 = features[labels == 0]
    features_1 = features[labels == 1]
    mu = np.array([np.mean(features_0, axis=0), np.mean(features_1, axis=0)])
    center_0 = features_0 - mu[0]
    center_1 = features_1 - mu[1]
    prior = np.array([features_0.shape[0] / (features_0.shape[0] + features_1.shape[0]), features_1.shape[0] / (features_0.shape[0] + features_1.shape[0])])

    covariance_0 = (np.transpose(center_0) @ center_0) / features_0.shape[0]
    covariance_1 = (np.transpose(center_1) @ center_1) / features_1.shape[0]
    covariance_shared = (features_0.shape[0] / (features_0.shape[0] + features_1.shape[0])) * covariance_0 + (features_1.shape[0] / (features_0.shape[0] + features_1.shape[0])) * covariance_1

    cov = covariance_shared

    return prior, mu, cov

def train_GDA_variable_variance(features, labels):
    prior, mu, cov = None, None, None
    features_0 = features[labels == 0]
    features_1 = features[labels == 1]
    mu = np.array([np.mean(features_0, axis=0), np.mean(features_1, axis=0)])
    center_0 = features_0 - mu[0]
    center_1 = features_1 - mu[1]
    prior = np.array([features_0.shape[0] / (features_0.shape[0] + features_1.shape[0]), features_1.shape[0] / (features_0.shape[0] + features_1.shape[0])])

    covariance_0 = (np.transpose(center_0) @ center_0) / features_0.shape[0]
    covariance_1 = (np.transpose(center_1) @ center_1) / features_1.shape[0]

    cov = (covariance_0, covariance_1)

    return prior, mu, cov
#**************************************************************************************************


#GDA testing and accuracy analyis functions
#**************************************************************************************************
#assign labels using trained GDA parameters for testing features
def predict(testing_features, theta, mu, cov):
    predicted_labels = None
    global gda_type
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, "testset")
    
    if isinstance(cov, tuple):
        gda_type = "GDA with variable variance"
        cov_0, cov_1 = cov  # Variable covariance - unpack tuple
        folder = "segmentation_GDA_variable"
    else:
        gda_type = "GDA with common variance"
        cov_0 = cov  # Common covariance - same matrix for both
        cov_1 = cov
        folder = "segmentation_GDA_common"

    output_path = os.path.join(base_dir, folder)
    os.makedirs(output_path, exist_ok=True)

    quadratic_0 = np.sum((testing_features - mu[0]) * np.transpose(np.linalg.solve(cov_0, np.transpose(testing_features - mu[0]))), axis=1)
    quadratic_1 = np.sum((testing_features - mu[1]) * np.transpose(np.linalg.solve(cov_1, np.transpose(testing_features - mu[1]))), axis=1)

    posterior_log_0 = np.log(theta[0]) - 0.5 * quadratic_0 - 0.5 * np.linalg.slogdet(cov_0)[1]
    posterior_log_1 = np.log(theta[1]) - 0.5 * quadratic_1 - 0.5 * np.linalg.slogdet(cov_1)[1]
    predicted_labels = (posterior_log_1 > posterior_log_0).astype(int)

    for root, _, files in os.walk(test_path):
        for name in files:
            if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            
            image = plt.imread(os.path.join(root, name))
            height, width = image.shape[0], image.shape[1]
            features_img = image.reshape(-1, 3)

            quadratic_0_img = np.sum((features_img - mu[0]) * np.transpose(np.linalg.solve(cov_0, np.transpose(features_img - mu[0]))), axis=1)
            quadratic_1_img = np.sum((features_img - mu[1]) * np.transpose(np.linalg.solve(cov_1, np.transpose(features_img - mu[1]))), axis=1)

            posterior_log_0_img = np.log(theta[0]) - 0.5 * quadratic_0_img - 0.5 * np.linalg.slogdet(cov_0)[1]
            posterior_log_1_img = np.log(theta[1]) - 0.5 * quadratic_1_img - 0.5 * np.linalg.slogdet(cov_1)[1]
            predicted_img = (posterior_log_1_img > posterior_log_0_img).astype(int)

            segmented_img = np.zeros((height, width, 3))
            predicted_img = predicted_img.reshape(height, width)
            segmented_img[predicted_img == 1] = [1.0, 0.0, 0.0]  # Red for barrel
            segmented_img[predicted_img == 0] = [0.0, 0.0, 0.0]  # Black for non-barrel

            save_path = os.path.join(output_path, f"{os.path.splitext(name)[0]}_segmented.png")
            plt.imsave(save_path, segmented_img)

    return predicted_labels

#print precision/call for both classes to console
#
#example console printout:
#GDA with common variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#GDA with variable variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#
def accuracy_analysis(predicted_labels, ground_truth_labels):
    global gda_type
    true_positive = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
    true_negative = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))
    false_positive = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
    false_negative = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))

    precision_0 = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
    precision_1 = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall_0 = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    recall_1 = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    #print(f"GDA with {'variable' if cov_0 is not cov_1 else 'common'} variance:")
    print(f"{gda_type}:")
    print(f"precision of label 0: {precision_0 * 100:.2f}%")
    print(f"recall of label 0:    {recall_0 * 100:.2f}%")
    print(f"precision of label 1: {precision_1 * 100:.2f}%")
    print(f"recall of label 1:    {recall_1 * 100:.2f}%")

    pass
#**************************************************************************************************
    

if __name__ == '__main__':
    #Please read this block before coding
    #**********************************************************************************************
    #caution: when you submit this file, make sure the main function is unchanged otherwise your
    #         grade will be affected because the grading script is designed based on the current
    #         main function
    #
    #         Also, do not print unnecessary values other than the accuracy analysis in the console

    #Labeling during runtime can be very time-consuming during the debugging phase. 
    #Also, it is hard to ensure the labelings are consistent during each testing run. 
    #Thus, we do this in separate stages.
    #First, implement all the functions and uncomment the three lines in the data loader block.
    #Then, revert the main function back to what it is used to be and start implementing the rest
    #**********************************************************************************************


    #data loader used to generate your labeling
    #ideally this block should only be called once
    #**********************************************************************************************
    # label_training_dataset('trainset', 'train_region')
    # label_testing_dataset('testset', 'test_region')
    #sys.exit(1)
    #**********************************************************************************************


    #import your generated labels from saved data
    #**********************************************************************************************
    training_features, training_labels = import_pre_labeled_training('trainset', 'train_region')
    testing_features, ground_truth_labels = import_pre_labeled_testing('testset', 'test_region')
    #**********************************************************************************************

    #GDA with common varianve
    #**********************************************************************************************
    prior, mu, cov = train_GDA_common_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************


    #GDA with variable variance
    #**********************************************************************************************
    prior, mu, cov = train_GDA_variable_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************
