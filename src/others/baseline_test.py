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
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import median_filter


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



# Function to get the custom colormap with the specified colors
def get_custom_colormap():
    colors = [
        '#000000',  # 0. BKG
        '#0000FF',  # 1. Asphalt
        '#0000AA',  # 2. Meadows
        '#00FFFF',  # 3. Gravel
        '#00FF00',  # 4. Trees
        '#FFFF00',  # 5. Painted metal sheets
        '#FF00FF',  # 6. Bare Soil
        '#FF0000',  # 7. Bitumen
        '#FFAA00',  # 8. Self-Blocking Bricks
        '#FFFFFF'   # 9. Shadows
    ]
    return ListedColormap(colors)


def visualize_ground_truth(ground_truth_map, dataset_name, colormap):
    plt.figure(figsize=(5, 5))
    plt.imshow(ground_truth_map, cmap=colormap)
    plt.axis('off')  # Hide axes
    os.makedirs('maps/ground_truth', exist_ok=True)
    filepath = os.path.join('maps/ground_truth', f"{dataset_name}_ground_truth_map.png")
    plt.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Ground truth map saved as {filepath}")




# Function to visualize classification map
def visualize_classification_map(classification_map, dataset_name, colormap):
    plt.figure(figsize=(5, 5))
    plt.imshow(classification_map, cmap=colormap)
    plt.axis('off')  # Hide axes
    os.makedirs('maps/1.baseline_GPU', exist_ok=True)
    filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_map.png")
    plt.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Classification map saved as {filepath}")




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





# Function to extract patches dynamically for classification
def extract_patch(image, row, col, patch_size=3):
    """
    Extract a patch centered at (row, col) from the image.
    
    Args:
        image (ndarray): Hyperspectral image (height x width x bands).
        row (int): Row index of the center pixel.
        col (int): Column index of the center pixel.
        patch_size (int): Size of the patch to extract.
        
    Returns:
        patch (ndarray): Flattened patch.
    """
    half_patch = patch_size // 2
    # Handle boundary conditions by padding the image
    padded_image = np.pad(image, ((half_patch, half_patch), (half_patch, half_patch), (0, 0)), mode='reflect')
    
    # Extract the patch centered at the pixel (row, col)
    start_row = row
    start_col = col
    patch = padded_image[start_row:start_row + patch_size, start_col:start_col + patch_size, :]
    
    return patch.flatten()

# Updated model_train_test function with sliding window patch extraction during inference
def model_train_test_with_patches(hsi_image, gt, patch_size=3, test_size=0.2, random_state=42):
    """
    Train and test the CatBoost classifier using pixel-level data but apply patch-based classification during inference.
    
    Args:
        hsi_image (ndarray): Hyperspectral image (height x width x bands).
        gt (ndarray): Ground truth labels (height x width).
        patch_size (int): Size of the patch to extract during inference.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Seed for reproducibility.
        
    Returns:
        oa (float): Overall accuracy.
        aa (float): Average accuracy.
        kappa (float): Kappa coefficient.
        training_time (float): Time taken to train the model.
        testing_time (float): Time taken to test the model.
        classification_map (ndarray): Predicted classification map (height x width).
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    
    # Flatten the ground truth labels for training and testing
    gt_reshaped = gt.reshape(-1)

    # Reshape the image into a 2D array for training (samples x bands)
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_reshaped, gt_reshaped, stratify=gt_reshaped, test_size=test_size, random_state=random_state
    )

    # Apply SMOTE to balance the class distribution in the training data
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Create and train the CatBoost classifier
    cbc = CatBoostClassifier(
        task_type='GPU',
        verbose=100
    )

    # Train the model
    start_train_time = time.time()
    cbc.fit(X_train_smote, y_train_smote)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Test the model on the pixel-based test data
    start_test_time = time.time()
    y_pred = cbc.predict(X_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)

    # Now classify the entire image using patches (sliding window approach)
    classification_map = np.zeros((hsi_image.shape[0], hsi_image.shape[1]), dtype=np.int32)
    
    for row in range(hsi_image.shape[0]):
        for col in range(hsi_image.shape[1]):
            patch = extract_patch(hsi_image, row, col, patch_size=patch_size)
            prediction = cbc.predict(patch.reshape(1, -1))
            classification_map[row, col] = prediction

    # Apply a median filter to the classification map to reduce noise
    classification_map_smooth = median_filter(classification_map, size=3)

    return oa, aa, kappa, training_time, testing_time, classification_map_smooth

# Example usage of the updated function for Pavia University
colormap = get_custom_colormap()

# Load dataset-Pavia university
# pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
# pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']

# # Train, test, and visualize for Pavia University with patch-based inference
# oa_pavia_u, aa_pavia_u, kappa_pavia_u, training_time_pavia_u, testing_time_pavia_u, classification_map_pavia_u = model_train_test_with_patches(pavia_u, pavia_u_gt, patch_size=3)

# # Visualize the classification map and ground truth map
# visualize_ground_truth(pavia_u_gt, "Pavia_University_gt", colormap)
# visualize_classification_map(classification_map_pavia_u, "Pavia_University", colormap)

# # Print metrics for Pavia University
# print(f"Pavia University - Overall Accuracy: {oa_pavia_u * 100:.2f}%")
# print(f"Pavia University - Average Accuracy: {aa_pavia_u * 100:.2f}%")
# print(f"Pavia University - Kappa Coefficient: {kappa_pavia_u:.4f}")
# print(f"Pavia University - Training Time: {training_time_pavia_u:.2f} sec")
# print(f"Pavia University - Testing Time: {testing_time_pavia_u:.2f} sec")


# Load dataset-Pavia centre
compi = scipy.io.loadmat('contents/data/compi.mat')['pavia']
compi_gt = scipy.io.loadmat('contents/data/compi_gt.mat')['pavia_gt']


# Train, test, and visualize for Pavia University with patch-based inference
oa_compi, aa_compi, kappa_compi, training_time_compi, testing_time_compi, classification_map_compi = model_train_test_with_patches(compi, compi_gt, patch_size=3)

# Visualize the classification map and ground truth map
visualize_ground_truth(compi_gt, "compientre_gt", colormap)
visualize_classification_map(classification_map_compi, "compientre", colormap)

# Print metrics for Pavia University
print(f"Pavia University - Overall Accuracy: {oa_compi * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa_compi * 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa_compi:.4f}")
print(f"Pavia University - Training Time: {training_time_compi:.2f} sec")
print(f"Pavia University - Testing Time: {testing_time_compi:.2f} sec")