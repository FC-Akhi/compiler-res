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
import os  # For directory operations
import numpy as np  # For numerical computations and array manipulations
import time  # To calculate training time

import scipy  # For scientific computations and loading datasets
from catboost import CatBoostClassifier  # For the CatBoost classifier model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For evaluation metrics
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
import matplotlib.pyplot as plt  # For plotting graphs and visualization
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets by oversampling
from collections import Counter  # To count occurrences of each class label
# importing the 'accuracy_score' function from the 'sklearn' library for evaluating classification accuracy
from sklearn.metrics import accuracy_score,  precision_recall_fscore_support  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay




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
def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
    """
    Visualize classification map and ground truth side by side and save as an image.
    Args:
        classification_map (ndarray): The predicted classification map.
        ground_truth_map (ndarray): The actual ground truth map.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    # Create a figure with two subplots for ground truth and classification map
    plt.figure(figsize=(10, 5))

    # Display ground truth map
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap='jet')
    plt.title(f"Ground Truth - {dataset_name}")
    plt.axis('off')  # Hide axes

    # Display classification map
    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap='jet')
    plt.title(f"Classification Map - {dataset_name}")
    plt.axis('off')  # Hide axes

    # Ensure the directory for saving the map exists, create it if it doesn't
    filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_vs_ground_truth.png")
    os.makedirs('maps/1.baseline_GPU', exist_ok=True)
    
    # Save the figure as an image
    plt.savefig(filepath)
    print(f"Figure saved as {dataset_name}_classification_vs_ground_truth.png")




# @brief: Calculate additional metrics such as confusion matrix, precision, recall, and F1-score per class.
# This function computes these metrics, prints them, and visualizes the confusion matrix.
def calculate_additional_metrics(y_true, y_pred):
    """
    Calculate and display confusion matrix, precision, recall, and F1-score per class.
    Args:
        y_true (ndarray): Ground truth labels for the test data.
        y_pred (ndarray): Predicted labels from the classifier.
    """
    # Compute the confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Display confusion matrix as a plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Display the classification report (precision, recall, F1-score, and support)
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred)
    print(report)

    # Calculate precision, recall, and F1-score per class without averaging
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Print additional metrics for each class
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("F1-Score per class:", fscore)




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
    

    # Train the CatBoost model on the SMOTE-applied training data
    start = time.time()
    cbc.fit(X_train_smote, y_train_smote)
    end = time.time()
    total_time = end - start  # Total training time

    # Test the trained model on the test set
    y_pred = cbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model on the test data

    # Display additional metrics like confusion matrix, precision, recall, and F1-score
    calculate_additional_metrics(y_test, y_pred)

    # Reshape the entire hyperspectral image (3D) to 2D for generating the classification map
    hsi_image_reshaped = hsi_image.reshape(-1, hsi_image.shape[2])

    # Generate class predictions for the entire hyperspectral image
    y_pred_full = cbc.predict(hsi_image_reshaped)

    # Reshape the predicted labels back to the original image dimensions (height x width)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Reshape the ground truth labels back to the original image dimensions (height x width)
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # params = cbc.get_all_params()
    # print("Full parameters:", params)

    # Return the accuracy, training time, classification map, and ground truth map
    return acc, total_time, classification_map, ground_truth_map






# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']

# # Analyze data set class distribution
# analyze_class_distribution(pavia_u_gt, 'pavia_u')


# Train, test, and visualize for Pavia University
acc_pavia_u, training_time_pavia_u, classification_map_pavia_u, ground_truth_map_pavia_u = model_train_test(pavia_u, pavia_u_gt)


# visualize classification map vs ground truth
visualize_classification_map(classification_map_pavia_u, ground_truth_map_pavia_u, "Pavia University")






# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']

# # Analyze data set class distribution
# analyze_class_distribution(pavia_c_gt, 'pavia_c')


# Train, test, and visualize for Pavia University
acc_pavia_c, training_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = model_train_test(pavia_c, pavia_c_gt)


# visualize classification map vs ground truth
visualize_classification_map(classification_map_pavia_c, ground_truth_map_pavia_c, "Pavia Centre")






# Load dataset-Salinas
salinas = scipy.io.loadmat('contents/data/Salinas.mat')['salinas']
salinas_gt = scipy.io.loadmat('contents/data/Salinas_gt.mat')['salinas_gt']

# # Analyze data set class distribution
# analyze_class_distribution(salinas_gt, 'salinas')


# Train, test, and visualize for Pavia University
acc_salinas, training_time_salinas, classification_map_salinas, ground_truth_map_salinas = model_train_test(salinas, salinas_gt)


# visualize classification map vs ground truth
visualize_classification_map(classification_map_salinas, ground_truth_map_salinas, "Salinas")




# # Load dataset-Indian Pines
# indian_pines = scipy.io.loadmat('contents/data/Indian_pines.mat')['indian_pines']
# indian_pines_gt = scipy.io.loadmat('contents/data/Indian_pines_gt.mat')['indian_pines_gt']

# # Analyze data set class distribution
# analyze_class_distribution(indian_pines_gt, 'indian_pines')


# # Train, test, and visualize for Pavia University
# acc_indian_pines, training_time_indian_pines, classification_map_indian_pines, ground_truth_map_indian_pines = model_train_test(indian_pines, indian_pines_gt)


# # visualize classification map vs ground truth
# visualize_classification_map(classification_map_indian_pines, ground_truth_map_indian_pines, "Indian Pines")





# Print accuracies and training times
print(f"Pavia University - Training Time: {training_time_pavia_u:.2f} sec, Accuracy: {acc_pavia_u * 100:.2f}%")
# Print accuracies and training times
print(f"Pavia Centre - Training Time: {training_time_pavia_c:.2f} sec, Accuracy: {acc_pavia_c * 100:.2f}%")
# Print accuracies and training times
print(f"Salinas - Training Time: {training_time_salinas:.2f} sec, Accuracy: {acc_salinas * 100:.2f}%")
# # Print accuracies and training times
# print(f"Indian Pines - Training Time: {training_time_indian_pines:.2f} sec, Accuracy: {acc_indian_pines * 100:.2f}%")





