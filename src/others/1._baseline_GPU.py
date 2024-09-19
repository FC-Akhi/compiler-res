"""
This is the implementation of CatBoost with SMOTE for data balancing.

Datasets:
- Pavia University (Pavia U)
- Pavia Centre (Pavia C)
- Salinas
- Indian Pines

Architecture:
- GPU-based implementation using CatBoost for classification.

Program Description:
- The code trains a CatBoost classifier on hyperspectral image data(all bands), incorporating the SMOTE technique to handle class imbalance.
- Classification maps are generated for visual comparison with ground truth data.
- The program evaluates the model's performance using metrics such as accuracy, precision, recall, and F1-score.

Authors:
- FC-Akhi, Nikhil Badoni
- Date: 14th Sept. 2024
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




# @brief: Analyze the class distribution of the ground truth labels and save the plot as an image.
# This function reshapes the ground truth, counts the number of pixels per class, and visualizes
# the class distribution as a bar plot. It also saves the plot in the specified directory.
def analyze_class_distribution(gt, dataset_name):
    """
    Analyze the class distribution of the ground truth labels and save the plot as an image.
    Args:
        gt (ndarray): Ground truth labels for the dataset.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    # Reshape the ground truth to a 1D array (flatten it)
    gt_reshaped = gt.reshape(-1)
    
    # Count the occurrence of each class label using Counter
    class_distribution = Counter(gt_reshaped)
    
    # Print the class distribution
    print("Class distribution (before balancing):")
    for class_label, count in class_distribution.items():
        print(f"Class {class_label}: {count} pixels")
    
    # Plot the class distribution as a bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue')
    plt.title(f"Class Distribution (Before Balancing) - {dataset_name}")
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Pixels")
    
    # Ensure the directory for saving the plot exists, create it if it doesn't
    filepath = os.path.join('class_dis_graph', f"{dataset_name}_class_distribution.png")
    os.makedirs('class_dis_graph', exist_ok=True)
    
    # Save the plot as an image in the specified directory
    plt.savefig(filepath)
    print(f"Class distribution plot saved as {filepath}")




# @brief: Visualize the classification map and ground truth side by side.
# This function displays the classification map and ground truth, then saves the result as an image.
def visualize_classification_map(classification_map, dataset_name):
    """
    Visualize and save only the classification map without a background.
    
    Args:
        classification_map (ndarray): The predicted classification map.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    # Create a figure for the classification map
    plt.figure(figsize=(5, 5))
    plt.imshow(classification_map, cmap='jet')  # Use 'jet' colormap for classification map
    plt.axis('off')  # Hide axes

    # Ensure the directory for saving the map exists
    os.makedirs('maps/1.baseline_GPU', exist_ok=True)
    filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_map.png")

    # Save the figure as an image with a transparent background and no padding
    plt.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Figure saved as {filepath}")




# Function to apply SMOTE for class balancing
def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote



# Function to calculate OA, AA, and Kappa
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




# Function to apply SMOTE
def apply_smote(X_train, y_train, random_state=42):
    """
    This function applies SMOTE to the training data to balance the class distribution.
    
    Inputs:
    - X_train: Training features
    - y_train: Training labels
    - random_state: Random state for reproducibility (default is 42)
    
    Outputs:
    - X_train_smote: SMOTE-resampled training features
    - y_train_smote: SMOTE-resampled training labels
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, y_train_smote


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

    # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        task_type='GPU',
        verbose=False
    )
    

   # Train the model
    start_train_time = time.time()
    cbc.fit(X_train_smote, y_train_smote)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Test the model
    start_test_time = time.time()
    y_pred = cbc.predict(X_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)


    # Reshape the entire hyperspectral image (3D) to 2D for generating the classification map
    hsi_image_reshaped = hsi_image.reshape(-1, hsi_image.shape[2])

    # Generate class predictions for the entire hyperspectral image
    y_pred_full = cbc.predict(hsi_image_reshaped)

    # Reshape the predicted labels back to the original image dimensions (height x width)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Reshape the ground truth labels back to the original image dimensions (height x width)
    # ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])


    # Return the calculated metrics and classification map
    return oa, aa, kappa, training_time, testing_time, classification_map





# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']


# Train, test, and visualize for Pavia University
oa_pavia_u, aa_pavia_u, kappa_pavia_u, training_time_pavia_u, testing_time_pavia_u, classification_map_pavia_u = model_train_test(pavia_u, pavia_u_gt)

# visualize classification map vs ground truth
visualize_classification_map(classification_map_pavia_u, "1.Pavia University")

# Print metrics for Pavia University
print(f"Pavia University - Overall Accuracy: {oa_pavia_u * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa_pavia_u * 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa_pavia_u:.4f}")
print(f"Pavia University - Training Time: {training_time_pavia_u:.2f} sec")
print(f"Pavia University - Testing Time: {testing_time_pavia_u:.2f} sec")




# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']


# Train, test, and visualize for Pavia Centre
oa_pavia_c, aa_pavia_c, kappa_pavia_c, training_time_pavia_c, testing_time_pavia_c, classification_map_pavia_c = model_train_test(pavia_c, pavia_c_gt)

# visualize classification map vs ground truth
visualize_classification_map(classification_map_pavia_c, "2.Pavia Centre")

# Print metrics for Pavia Centre
print(f"Pavia Centre - Overall Accuracy: {oa_pavia_c * 100:.2f}%")
print(f"Pavia Centre - Average Accuracy: {aa_pavia_c * 100:.2f}%")
print(f"Pavia Centre - Kappa Coefficient: {kappa_pavia_c:.4f}")
print(f"Pavia Centre - Training Time: {training_time_pavia_c:.2f} sec")
print(f"Pavia Centre - Testing Time: {testing_time_pavia_c:.2f} sec")










