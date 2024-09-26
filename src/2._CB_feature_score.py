"""
This is the implementation of CatBoost with SMOTE for data balancing and extracting feature scores.

Datasets:
- Pavia University (Pavia U)
- Pavia Centre (Pavia C)


Architecture:
- GPU-based implementation using CatBoost for classification.

Program Description:
- The code trains a CatBoost classifier on hyperspectral image data(all bands), incorporating the SMOTE technique to handle class imbalance.
- Feature scores are generated & high feature score bands are taken for model train 
- Classification maps are generated for visual comparison with ground truth data.
- The program evaluates the model's performance using metrics such as accuracy, precision, recall, and F1-score.

Authors:
- FC-Akhi Nikhil Badoni
- Date: 21st Sept. 2024
"""



# Import necessary libraries
import os
import numpy as np
import time
import scipy
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import median_filter

# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']

# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']


# Assuming this is your function for metrics calculation
def calculate_metrics(y_true, y_pred):
    """
    Calculate Overall Accuracy (OA), Average Accuracy (AA), and Kappa metrics.
    Args:
        y_true (ndarray): Ground truth labels.
        y_pred (ndarray): Predicted labels.
    Returns:
        oa: Overall Accuracy
        aa: Average Accuracy
        kappa: Kappa Coefficient
    """
    # Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy (diagonal values divided by total class samples)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Average Accuracy (AA)
    aa = np.mean(per_class_acc)

    # Kappa Coefficient
    kappa = cohen_kappa_score(y_true, y_pred)

    return oa, aa, kappa



# Dataset classification
def model_train_test(hsi_image, gt, test_size=0.2, random_state=42):
    """
    Method to create and train a CatBoost model using hyperspectral imagery data.
    The method will also apply SMOTE for balancing classes, test the model, and generate a classification map.
    
    Input:
    - hsi_image: 3D array of hyperspectral image data (height x width x bands)
    - gt: 2D array of ground truth labels (height x width)
    - test_size: Proportion of data to be used for testing (default = 0.2)
    - random_state: Random seed for reproducibility (default = 42)
    
    Output:
    - acc: Accuracy score of the model on the test data
    - total_time: Time taken to train the model
    - classification_map: Predicted class labels for the entire hyperspectral image (2D array)
    - ground_truth_map: Ground truth labels (2D array)
    """

    # debug
    print("model_train_test start!!!")
    

    # Get the number of samples in the image (height x width)
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]

    # Get the number of spectral bands in the image (depth)
    n_bands = hsi_image.shape[2]

    # Reshape the 3D hyperspectral image (height x width x bands) to a 2D array (samples x bands)
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)
    
    
    print('Reshaping done', flush=True)
    print('\n', flush=True)

    # Split the reshaped data into training and testing sets, stratified by the ground truth labels
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_reshaped, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state
    )


    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_strength=2,
        l2_leaf_reg=1,
        task_type='CPU',
        early_stopping_rounds=50,
        verbose=50
    )
    

   # Train the model
    start_train_time = time.time()
    cbc.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Test the model
    start_test_time = time.time()
    y_pred = cbc.predict(X_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time


    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)



    # Return the calculated metrics and classification map
    return oa, aa, kappa, training_time, testing_time, cbc



def get_feature_scores(hsi_image, gt):
    """
    Method for extracting feature scores for the spectral bands in a hyperspectral image.
    
    Input:
    - hsi_image: 3D array of hyperspectral image data (height x width x bands)
    - gt: Ground truth labels (height x width)
    
    Output:
    - feature_score: Feature importance scores of each band in the hyperspectral image
    """

    # debug
    print("Feature score extraction started!!!")
    

    # Train the model and get the CatBoost model instance
    oa, aa, kappa, training_time, testing_time, cbc = model_train_test(hsi_image, gt)

    # Get feature importances using the 'PredictionValuesChange' method
    feature_score = cbc.get_feature_importance(type='PredictionValuesChange')

    print("Feature score generation done")

    # Return the feature score
    return oa, aa, kappa, training_time, testing_time, feature_score






oa, aa, kappa, training_time, testing_time, pavia_u_feature_scores = get_feature_scores(pavia_u, pavia_u_gt)
print(f"Pavia University - Overall Accuracy: {oa * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa * 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa:.4f}")
print(f"Pavia University - Training Time: {training_time:.2f} sec")
print(f"Pavia University - Testing Time: {testing_time:.2f} sec")


print(f"Feature_scores_pavia_u: {pavia_u_feature_scores}")







