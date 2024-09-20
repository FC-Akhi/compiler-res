import os  # For filesystem operations
import numpy as np  # For numerical computations
import time
import scipy  # For scientific computations and loading datasets
from catboost import CatBoostClassifier  # CatBoost classifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
import matplotlib.pyplot as plt  # For plotting graphs and visualization
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets by oversampling
from collections import Counter  # To count occurrences of each class label
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle  # For saving and loading models or objects

# Load Pavia University dataset
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']

# Repeat for Pavia Centre dataset
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']




# Function to calculate metrics (OA, AA, Kappa)
def calculate_metrics(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_true, y_pred)
    return oa, aa, kappa



# Function to apply SMOTE for class balancing
def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote



# Model training and testing with SMOTE integration
def model_train_test(hsi_image, gt, hsi_image_limited, test_size=0.2, random_state=42):

    gt_reshaped = gt.reshape(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        hsi_image_limited, gt_reshaped, stratify=gt_reshaped, test_size=test_size, random_state=random_state)
    
    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)
    
    # Create and train CatBoost classifier
    cbc = CatBoostClassifier(
        n_estimators=1500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_strength=2,
        l2_leaf_reg=1,
        task_type='GPU',
        early_stopping_rounds=50,
        verbose=100)
    
    start = time.time()
    cbc.fit(X_train_smote, y_train_smote)
    end = time.time()
    training_time = end - start

    # Testing phase
    start = time.time()
    y_pred = cbc.predict(X_test)
    end = time.time()
    testing_time = end - start
    # acc = accuracy_score(y_test, y_pred)

    oa, aa, kappa = calculate_metrics(y_test, y_pred)

    # Generate classification map
    y_pred_full = cbc.predict(hsi_image_limited)
    
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    return oa, aa, kappa, training_time, testing_time, cbc




# # Function to apply ICA for dimensionality reduction
# def ICA(hsi_image, gt):
#     n_samples = hsi_image.shape[0] * hsi_image.shape[1]
#     n_bands = hsi_image.shape[2]
#     hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

#     # Standardize the data
#     scaler = StandardScaler()
#     hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

#     # Apply ICA
#     ica = FastICA(random_state=42)
#     hsi_image_limited = ica.fit_transform(hsi_image_scaled)

#     # Print the number of bands before and after applying ICA
#     print(f"Number of bands before applying ICA: {n_bands}")
#     print(f"Number of components after applying ICA: {hsi_image_limited.shape[1]}")  # Print remaining components


#     # Train the model
#     oa, aa, kappa, total_time1, total_time2, cbc = model_train_test(hsi_image, gt, hsi_image_limited)
    
#     return oa, aa, kappa, total_time1, total_time2, cbc, hsi_image_limited





# Function to apply ICA for dimensionality reduction
def ICA(hsi_image, gt, n_components=51):
    """
    Apply ICA for dimensionality reduction on hyperspectral image.
    
    Args:
        hsi_image (ndarray): The hyperspectral image.
        gt (ndarray): Ground truth labels.
        n_components (int): Number of ICA components to reduce to (default 50).
        
    Returns:
        oa (float): Overall Accuracy.
        aa (float): Average Accuracy.
        kappa (float): Kappa score.
        total_time1 (float): Training time.
        total_time2 (float): Testing time.
        cbc (CatBoostClassifier): Trained CatBoost classifier.
        hsi_image_limited (ndarray): Reduced hyperspectral image.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Standardize the data
    scaler = StandardScaler()
    hsi_image_scaled = scaler.fit_transform(hsi_image_reshaped)

    # Apply ICA with n_components=50
    ica = FastICA(n_components=n_components, random_state=42)
    hsi_image_limited = ica.fit_transform(hsi_image_scaled)

    # Print the number of bands before and after applying ICA
    print(f"Number of bands before applying ICA: {n_bands}")
    print(f"Number of components after applying ICA: {hsi_image_limited.shape[1]}")  # Should print 50

    # Train the model
    oa, aa, kappa, total_time1, total_time2, cbc = model_train_test(hsi_image, gt, hsi_image_limited)
    
    return oa, aa, kappa, total_time1, total_time2, cbc, hsi_image_limited





# Train and test using ICA
oa_u, aa_u, kappa_u, training_time_pavia_u, testing_time_pavia_u, cbc_u, hsi_image_ica_transformed_u = ICA(pavia_u, pavia_u_gt)


# Print the accuracy metrics
# Print metrics for Pavia University
print(f"Pavia University - Overall Accuracy: {oa_u * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa_u* 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa_u:.4f}")
print(f"Pavia University - Testing Time: {testing_time_pavia_u:.2f} sec")



# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_u.save_model('CB-Fast_ICA_model_U.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('CB-Fast_ICA_reduced_band_combination_u.pkl', 'wb') as f:
    pickle.dump(hsi_image_ica_transformed_u, f)







oa_c, aa_c, kappa_c, training_time_pavia_c, testing_time_pavia_c, cbc_c, hsi_image_ica_transformed_c = ICA(pavia_c, pavia_c_gt)

# Print metrics for Pavia Centre
print(f"Pavia Centre - Overall Accuracy: {oa_c * 100:.2f}%")
print(f"Pavia Centre - Average Accuracy: {aa_c * 100:.2f}%")
print(f"Pavia Centre - Kappa Coefficient: {kappa_c:.4f}")
print(f"Pavia Centre - Testing Time: {testing_time_pavia_c:.2f} sec")

# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_c.save_model('CB-Fast_ICA_model_C.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('CB-Fast_ICA_reduced_band_combination_c.pkl', 'wb') as f:
    pickle.dump(hsi_image_ica_transformed_c, f)