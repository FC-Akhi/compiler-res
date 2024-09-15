
import os  # importing the 'os' module to work with the operating system
import numpy as np  # importing the 'numpy' library and renaming it to 'np'
import random  # importing random
import scipy  # importing the 'scipy' library for scientific computing
from catboost import CatBoostClassifier
import time  # importing the 'time' module to measure time
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap


# @brief: Visualize the classification map and ground truth side by side.
def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
    """
    Visualize classification map and ground truth side by side and save as an image.
    Args:
        classification_map (ndarray): The predicted classification map.
        ground_truth_map (ndarray): The actual ground truth map.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    # Ensure that the input maps are not changed
    classification_map = np.copy(classification_map)
    ground_truth_map = np.copy(ground_truth_map)

    # Create a figure with two subplots for ground truth and classification map
    plt.figure(figsize=(10, 5))

    # Display ground truth map (ensure no alteration to ground truth data)
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap='jet')  # Use 'jet' colormap for ground truth
    plt.title(f"Ground Truth - {dataset_name}")
    plt.axis('off')  # Hide axes

    # Display classification map (ensure no alteration to classification data)
    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap='jet')  # Use 'jet' colormap for classification map
    plt.title(f"Classification Map - {dataset_name}")
    plt.axis('off')  # Hide axes

    # Save the figure as an image
    os.makedirs('m', exist_ok=True)
    filepath = os.path.join('m', f"{dataset_name}_classification_vs_ground_truth.png")
    plt.savefig(filepath)
    print(f"Figure saved as {filepath}")
# Revised code (with mentioned changes)
def model_train_test(hsi_image, gt, hsi_image_limited, test_size=0.2, random_state=42):
    """
    Train CatBoost model using hsi_image_limited and return accuracy, training time, and classification maps.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    
    # Reshape ground truth to a 1D array for stratification
    gt_reshaped = gt.reshape(-1)

    # Split the reshaped data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_limited, gt_reshaped, stratify=gt_reshaped, test_size=test_size, random_state=random_state)

    # Further split training data to create a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        task_type='CPU',
        verbose=False,
        random_seed=random_state  # Ensure reproducibility
    )

    # Train the model with validation data for early stopping
    start = time.time()
    cbc.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)
    end = time.time()
    total_time = end - start

    # Test the model
    y_pred = cbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Generate classification map for the entire dataset
    y_pred_full = cbc.predict(hsi_image_limited)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(gt_reshaped, y_pred_full))

    print("Confusion Matrix:")
    print(confusion_matrix(gt_reshaped, y_pred_full))

    # Return accuracy, training time, and classification maps
    return acc, total_time, classification_map, ground_truth_map




def ICA(hsi_image, gt, n_components=20):
    """
    Apply ICA to reduce dimensions of hsi_image and train CatBoost model.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Standardize the data
    scaler = StandardScaler()
    hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

    # Ensure the number of components does not exceed the number of bands
    if n_components > n_bands:
        print(f"Reducing number of components to {n_bands} (number of bands).")
        n_components = n_bands

    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    hsi_image_limited = ica.fit_transform(hsi_image_scaled)

    # Train the CatBoost model using the reduced data
    acc, total_time, classification_map, ground_truth_map = model_train_test(hsi_image, gt, hsi_image_limited)

    return acc, total_time, classification_map, ground_truth_map




# Train, test, and visualize for Salinas
acc_pavia_c, training_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = ICA(pavia_c, pavia_c_gt)
visualize_classification_map(classification_map_pavia_c, ground_truth_map_pavia_c, "pavia_c")

print(f"pavia_c - Training Time: {training_time_pavia_c:.2f} sec, Accuracy: {acc_pavia_c * 100:.2f}%")
