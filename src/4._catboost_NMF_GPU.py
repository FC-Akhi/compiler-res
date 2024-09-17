"""This is the implementation of catBoost + MNF for band reduction technique
Datasets:
Pavia U
Pavia C
Salinas
Indian pines

Architecture: GPU"""


import os
import numpy as np
import scipy
from catboost import CatBoostClassifier
import time
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
    """
    Visualize classification map and ground truth side by side and save as an image.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap='jet')
    plt.title(f"Ground Truth - {dataset_name}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap='jet')
    plt.title(f"Classification Map - {dataset_name}")
    plt.axis('off')

    # Save the figure instead of showing it
    plt.savefig(f"{dataset_name}_classification_vs_ground_truth.png")
    print(f"Figure saved as {dataset_name}_classification_vs_ground_truth.png")




# Visualization Function
def visualize_classification_map(classification_map, dataset_name):
    """
    Visualize and save only the classification map without a background.
    
    Args:
        classification_map (ndarray): The predicted classification map.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(classification_map, cmap='jet')  # Use 'jet' colormap for classification map
    plt.axis('off')  # Hide axes

    # Ensure the directory for saving the map exists
    os.makedirs('maps/NMF', exist_ok=True)
    filepath = os.path.join('maps', f"{dataset_name}_classification_map.png")

    # Save the figure as an image with a transparent background and no padding
    plt.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Figure saved as {filepath}")


# Function to apply SMOTE
def apply_smote(X_train, y_train, random_state=42):
    """
    This function applies SMOTE to the training data to balance the class distribution.
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


# Model Training with SMOTE and CatBoost
def model_train_test(hsi_image, gt, hsi_image_limited, test_size=0.2, random_state=42):
    """
    Train CatBoost model using hsi_image_limited and return accuracy, training time, and classification maps.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    
    # Reshape ground truth to a 1D array for stratification
    gt_reshaped = gt.reshape(-1)

    # Split the reshaped data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_limited, gt_reshaped, stratify=gt_reshaped, test_size=test_size, random_state=random_state)

    # Apply SMOTE to balance the class distribution in the training data
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        iterations=1000,
        depth=6,
        loss_function='MultiClass',
        task_type='GPU',
        early_stopping_rounds=50,
        verbose=False
    )

    # Train the model
    start = time.time()
    cbc.fit(X_train_smote, y_train_smote)
    end = time.time()
    total_time1 = end - start

    # Test the model
    start = time.time()
    y_pred = cbc.predict(X_test)
    end = time.time()
    total_time2 = end - start
    acc = accuracy_score(y_test, y_pred)

    # Generate classification map for the entire dataset
    y_pred_full = cbc.predict(hsi_image_limited)

    # Reshape predictions and ground truth for visualization
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(gt_reshaped, y_pred_full))

    print("Confusion Matrix:")
    print(confusion_matrix(gt_reshaped, y_pred_full))

    # Return accuracy, training time, and classification maps
    return acc, total_time1, total_time2, classification_map, ground_truth_map


# NMF for Dimensionality Reduction
def NMF_transform(hsi_image, gt, n_components=20):
    """
    Apply NMF to reduce dimensions of hsi_image and train CatBoost model.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Apply MinMaxScaler to ensure all data is non-negative
    scaler = MinMaxScaler()
    hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

    # Ensure the number of components does not exceed the number of bands
    if n_components > n_bands:
        print(f"Reducing number of components to {n_bands} (number of bands).")
        n_components = n_bands

    # Apply NMF
    nmf = NMF(n_components=n_components, random_state=42, init='random', max_iter=500)
    hsi_image_limited = nmf.fit_transform(hsi_image_scaled)

    # Train the CatBoost model using the reduced data
    acc, total_time1, total_time2, classification_map, ground_truth_map = model_train_test(hsi_image, gt, hsi_image_limited)

    return acc, total_time1, total_time2, classification_map, ground_truth_map


# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']


# Train, test, and visualize for Pavia University
acc_pavia_u, training_time_pavia_u, testing_time_pavia_u, classification_map_pavia_u, ground_truth_map_pavia_u = NMF_transform(pavia_u, pavia_u_gt)
visualize_classification_map(classification_map_pavia_u, "pavia_u")

print(f"pavia_u - Training Time: {training_time_pavia_u:.2f} sec, Testing Time: {testing_time_pavia_u:.2f} sec, Accuracy: {acc_pavia_u * 100:.2f}%")



# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']

# Train, test, and visualize for Pavia Centre
acc_pavia_c, training_time_pavia_c, testing_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = NMF_transform(pavia_c, pavia_c_gt)
visualize_classification_map(classification_map_pavia_c, "pavia_c")

print(f"pavia_c - Training Time: {training_time_pavia_c:.2f} sec, Testing Time: {testing_time_pavia_c:.2f} sec, Accuracy: {acc_pavia_c * 100:.2f}%")


