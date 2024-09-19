"""This is the implementation of catBoost + ICA for band reduction technique
Datasets:
Pavia U
Pavia C
Salinas
Indian pines

Architecture: GPU"""


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



# # @brief: Visualize the classification map and ground truth side by side.
# def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
#     """
#     Visualize classification map and ground truth side by side and save as an image.
#     Args:
#         classification_map (ndarray): The predicted classification map.
#         ground_truth_map (ndarray): The actual ground truth map.
#         dataset_name (str): Name of the dataset to use for file naming.
#     """
#     # Create a figure with two subplots for ground truth and classification map
#     plt.figure(figsize=(10, 5))

#     # Display ground truth map
#     plt.subplot(1, 2, 1)
#     plt.imshow(ground_truth_map, cmap='jet')
#     plt.title(f"Ground Truth - {dataset_name}")
#     plt.axis('off')  # Hide axes

#     # Display classification map
#     plt.subplot(1, 2, 2)
#     plt.imshow(classification_map, cmap='jet')
#     plt.title(f"Classification Map - {dataset_name}")
#     plt.axis('off')  # Hide axes

#     # Ensure the directory for saving the map exists, create it if it doesn't
#     os.makedirs('maps/1.baseline_GPU', exist_ok=True)
#     filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_vs_ground_truth.png")
    
#     # Save the figure as an image
#     plt.savefig(filepath)
#     print(f"Figure saved as {filepath}")




# def model_train_test(hsi_image, gt, hsi_image_limited, test_size=0.2, random_state=42):
#     """
#     Train CatBoost model using hsi_image_limited and return accuracy, training time, and classification maps.
#     """

#     # Reshape the image into a 2D array of samples and bands
#     n_samples = hsi_image.shape[0] * hsi_image.shape[1]
#     n_bands = hsi_image.shape[2]
#     hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

#     # Split the reshaped data into train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         hsi_image_limited, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)

    

#     # Create CatBoost classifier
#     cbc = CatBoostClassifier(
#         task_type='GPU',
#         verbose=False
#     )

#     # Train the model (include early stopping with test set as validation)
#     start = time.time()
#     cbc.fit(X_train, y_train, eval_set=(X_test, y_test))
#     end = time.time()
#     total_time = end - start

#     # Test the model
#     y_pred = cbc.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)

#     # Debugging: Print accuracy on the test set
#     print(f"Test set accuracy: {acc}")

#     # Generate classification map for the entire dataset
#     y_pred_full = cbc.predict(hsi_image_limited)

#     # Debugging: Check unique classes in the predicted full map
#     print(f"Unique classes in full prediction: {np.unique(y_pred_full)}")

#     # Reshape predictions and ground truth for visualization
#     classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
#     ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

#     # Debugging: Check shape and values of classification and ground truth maps
#     print(f"Shape of classification map: {classification_map.shape}")
#     print(f"Shape of ground truth map: {ground_truth_map.shape}")
#     print(f"Unique values in classification map: {np.unique(classification_map)}")
#     print(f"Unique values in ground truth map: {np.unique(ground_truth_map)}")

#     # Print classification report and confusion matrix
#     print("Classification Report:")
#     print(classification_report(gt.reshape(-1), y_pred_full))

#     print("Confusion Matrix:")
#     print(confusion_matrix(gt.reshape(-1), y_pred_full))

#     # Return accuracy, training time, and classification maps
#     return acc, total_time, classification_map, ground_truth_map


# def ICA(hsi_image, gt, n_components=20):
#     """
#     Apply ICA to reduce dimensions of hsi_image and train CatBoost model.
#     """
#     # Reshape hyperspectral image (height, width, bands) into (samples, bands)
#     n_samples = hsi_image.shape[0] * hsi_image.shape[1]
#     n_bands = hsi_image.shape[2]
#     hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

#     # Standardize the data
#     scaler = StandardScaler()
#     hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

#     if n_components > n_bands:
#         raise ValueError("Number of components exceeds the number of bands in the image.")

#     # Apply ICA
#     ica = FastICA(n_components=n_components, random_state=42)
#     hsi_image_limited = ica.fit_transform(hsi_image_scaled)

#     # Debugging: Check the shape of the reduced data
#     print(f"Shape of data after ICA reduction: {hsi_image_limited.shape}")

#     # Train the CatBoost model using the reduced data
#     acc, total_time, classification_map, ground_truth_map = model_train_test(hsi_image, gt, hsi_image_limited)

#     # Return the results
#     return acc, total_time, classification_map, ground_truth_map



from matplotlib.colors import ListedColormap





# Define a consistent colormap for both classification map and ground truth
def create_custom_colormap():
    # Create a custom colormap with strong colors (distinct for each class)
    colors = [
        '#FF0000',  # red
        '#00FF00',  # green
        '#0000FF',  # blue
        '#FFFF00',  # yellow
        '#FF00FF',  # magenta
        '#00FFFF',  # cyan
        '#800000',  # maroon
        '#808000',  # olive
        '#008080',  # teal
        '#800080',  # purple
        '#FF4500',  # orange red
        '#2E8B57',  # sea green
        '#4682B4',  # steel blue
        '#DAA520',  # golden rod
        '#A52A2A',  # brown
        '#DEB887',  # burlywood
        '#D2691E',  # chocolate
    ]
    return ListedColormap(colors)

# Custom colormap for classification and ground truth
custom_cmap = create_custom_colormap()

def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
    """
    Visualize classification map and ground truth side by side and save as an image.
    Args:
        classification_map (ndarray): The predicted classification map.
        ground_truth_map (ndarray): The actual ground truth map.
        dataset_name (str): Name of the dataset to use for file naming.
    """
    plt.figure(figsize=(10, 5))

    # Display ground truth map with consistent colormap
    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap=custom_cmap)
    plt.title(f"Ground Truth - {dataset_name}")
    plt.axis('off')

    # Display classification map with the same colormap
    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap=custom_cmap)
    plt.title(f"Classification Map - {dataset_name}")
    plt.axis('off')

    # Ensure the directory for saving the map exists, create it if it doesn't
    os.makedirs('maps/1.baseline_GPU', exist_ok=True)
    filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_vs_ground_truth.png")

    # Save the figure as an image
    plt.savefig(filepath)
    print(f"Figure saved as {filepath}")

def model_train_test(hsi_image, gt, hsi_image_limited, test_size=0.2, random_state=42):
    """
    Train CatBoost model using hsi_image_limited and return accuracy, training time, and classification maps.
    """

    # Reshape the image into a 2D array of samples and bands
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Split the reshaped data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_limited, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)

    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        task_type='GPU',
        verbose=False
    )

    # Train the model (include early stopping with test set as validation)
    start = time.time()
    cbc.fit(X_train, y_train, eval_set=(X_test, y_test))
    end = time.time()
    total_time = end - start

    # Test the model
    y_pred = cbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Generate classification map for the entire dataset
    y_pred_full = cbc.predict(hsi_image_limited)

    # Reshape predictions and ground truth for visualization
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(gt.reshape(-1), y_pred_full))

    print("Confusion Matrix:")
    print(confusion_matrix(gt.reshape(-1), y_pred_full))

    # Return accuracy, training time, and classification maps
    return acc, total_time, classification_map, ground_truth_map

def ICA(hsi_image, gt, n_components=20):
    """
    Apply ICA to reduce dimensions of hsi_image and train CatBoost model.
    """
    # Reshape hyperspectral image (height, width, bands) into (samples, bands)
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Standardize the data
    scaler = StandardScaler()
    hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

    if n_components > n_bands:
        raise ValueError("Number of components exceeds the number of bands in the image.")

    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    hsi_image_limited = ica.fit_transform(hsi_image_scaled)

    # Train the CatBoost model using the reduced data
    acc, total_time, classification_map, ground_truth_map = model_train_test(hsi_image, gt, hsi_image_limited)

    # Return the results
    return acc, total_time, classification_map, ground_truth_map















# Load datasets
# pavia_u = scipy.io.loadmat('src2/PaviaU.mat')['paviaU']
# pavia_u_gt = scipy.io.loadmat('src2/PaviaU_gt.mat')['paviaU_gt']

# pavia_c = scipy.io.loadmat('src2/Pavia.mat')['pavia']
# pavia_c_gt = scipy.io.loadmat('src2/Pavia_gt.mat')['pavia_gt']

salinas = scipy.io.loadmat('contents/data/Salinas.mat')['salinas']
salinas_gt = scipy.io.loadmat('contents/data/Salinas_gt.mat')['salinas_gt']

indian_pines = scipy.io.loadmat('contents/data/Indian_pines.mat')['indian_pines']
indian_pines_gt = scipy.io.loadmat('contents/data/Indian_pines_gt.mat')['indian_pines_gt']




# Train, test, and visualize for Pavia University
# acc_pavia_u, training_time_pavia_u, classification_map_pavia_u, ground_truth_map_pavia_u = ICA(pavia_u, pavia_u_gt)
# visualize_classification_map(classification_map_pavia_u, ground_truth_map_pavia_u, "Pavia University")

# # Train, test, and visualize for Pavia Centre
# acc_pavia_c, training_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = ICA(pavia_c, pavia_c_gt)
# visualize_classification_map(classification_map_pavia_c, ground_truth_map_pavia_c, "Pavia Centre")

# Train, test, and visualize for Salinas
acc_salinas, training_time_salinas, classification_map_salinas, ground_truth_map_salinas = ICA(salinas, salinas_gt)
visualize_classification_map(classification_map_salinas, ground_truth_map_salinas, "Salinas")

# Train, test, and visualize for Indian Pines
acc_indian_pines, training_time_indian_pines, classification_map_indian_pines, ground_truth_map_indian_pines = ICA(indian_pines, indian_pines_gt)
visualize_classification_map(classification_map_indian_pines, ground_truth_map_indian_pines, "Indian Pines")




# Print accuracies and training times
# print(f"Pavia University - Training Time: {training_time_pavia_u:.2f} sec, Accuracy: {acc_pavia_u * 100:.2f}%")
# print(f"Pavia Centre - Training Time: {training_time_pavia_c:.2f} sec, Accuracy: {acc_pavia_c * 100:.2f}%")
print(f"Salinas - Training Time: {training_time_salinas:.2f} sec, Accuracy: {acc_salinas * 100:.2f}%")
print(f"Indian Pines - Training Time: {training_time_indian_pines:.2f} sec, Accuracy: {acc_indian_pines * 100:.2f}%")