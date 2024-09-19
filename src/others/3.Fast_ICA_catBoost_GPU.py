from imblearn.over_sampling import SMOTE
import os  # importing the 'os' module to work with the operating system
import numpy as np  # importing the 'numpy' library and renaming it to 'np'
import time
import scipy  # importing the 'scipy' library for scientific computing
import scipy  # For scientific computations and loading datasets
from catboost import CatBoostClassifier  # For the CatBoost classifier model
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
import matplotlib.pyplot as plt  # For plotting graphs and visualization
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets by oversampling
from collections import Counter  # To count occurrences of each class label
# importing the 'accuracy_score' function from the 'sklearn' library for evaluating classification accuracy
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


# @brief: Visualize the classification map and ground truth side by side.
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
    os.makedirs('maps/3.ICA', exist_ok=True)
    filepath = os.path.join('maps/3.ICA', f"{dataset_name}_classification_map.png")

    # Save the figure as an image with a transparent background and no padding
    plt.savefig(filepath, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Figure saved as {filepath}")


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
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote



# Revised model_train_test with SMOTE integration
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

    # Return accuracy, training time, and classification maps
    return acc, total_time1, total_time2, classification_map, ground_truth_map, cbc



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
    acc, total_time1, total_time2, classification_map, ground_truth_map, cbc = model_train_test(hsi_image, gt, hsi_image_limited)

    return acc, total_time1, total_time2, classification_map, ground_truth_map, cbc







# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']


# Train, test, and visualize for pavia university
acc_pavia_u, training_time_pavia_u, testing_time_pavia_u, classification_map_pavia_u, ground_truth_map_pavia_u, cbc = ICA(pavia_u, pavia_u_gt)



# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']

# Train, test, and visualize for pavia centre
acc_pavia_c, training_time_pavia_c, testing_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = ICA(pavia_c, pavia_c_gt)
visualize_classification_map(classification_map_pavia_c, ground_truth_map_pavia_c, "pavia_c")

print(f"pavia_c - Training Time: {training_time_pavia_c:.2f} sec,  test Time: {testing_time_pavia_c:.2f} sec,Accuracy: {acc_pavia_c * 100:.2f}%")





def classification_map_final(hsi_image, gt, test_size = 0.2, random_state=42, cbc=None):

    """
    Input: band combination, original hsi image with its gt, original feature score list of cooresponding image
    - In this part band combination will received
    - And then delete rest of the bands from image keeping only the combination
    - Pass to  catBoost
    Output:
    - Returns the accuracy after using input band_combination"""

    n_samples = hsi_image.shape[0] * hsi_image.shape[1] # get the number of samples in the image
    n_bands = hsi_image.shape[2] # get the number of bands in the image
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands) # reshape the image into a 2D array of samples and bands


    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reshaped, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)  # split the data into training and testing sets
    

    # # Train the model
    
    cbc.fit(X_train, y_train)  # train the classifier on the training set
    
    
    
    # Test the model
    start_test_time = time.time()
    y_pred = cbc.predict(X_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)


    # Generate classification map for the entire dataset
    y_pred_full = cbc.predict(hsi_image_reshaped)
    
    
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    


    # Return accuracy, training time, and classification maps
    return oa, aa, kappa, testing_time, classification_map






oa_pavia_u, aa_pavia_u, kappa_pavia_u, testing_time_pavia_u, classification_map_pavia_u  = classification_map_final(pavia_u, pavia_u_gt, test_size = 0.2, random_state=42, cbc=cbc)

visualize_classification_map(classification_map_pavia_u, f"1. Pavia_University_map_final")

# Print metrics for Pavia University
print(f"Pavia University - Overall Accuracy: {oa_pavia_u * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa_pavia_u * 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa_pavia_u:.4f}")
print(f"Pavia University - Testing Time: {testing_time_pavia_u:.2f} sec")



oa_pavia_c, aa_pavia_c, kappa_pavia_c, testing_time_pavia_c, classification_map_pavia_c = classification_map_final(solutions_pavia_c[0][0], pavia_c, pavia_c_gt, test_size = 0.2, random_state=42, cbc=cbc)

visualize_classification_map(classification_map_pavia_c, f"2. Pavia_Centre_map_final")

# Print metrics for Pavia Centre
print(f"Pavia Centre - Overall Accuracy: {oa_pavia_c * 100:.2f}%")
print(f"Pavia Centre - Average Accuracy: {aa_pavia_c * 100:.2f}%")
print(f"Pavia Centre - Kappa Coefficient: {kappa_pavia_c:.4f}")
print(f"Pavia Centre - Testing Time: {testing_time_pavia_c:.2f} sec")

