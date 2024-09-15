import numpy as np
import scipy.io
from catboost import CatBoostClassifier
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pymrmr
import pandas as pd


# KL Divergence calculation
def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    
    # Avoid log(0) and division by zero
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return np.sum(p * np.log(p / q))


# Band selection using KL divergence and mRMR
def band_selection_kl_mrmr(hsi_image, labels, num_bands):
    """
    Perform band selection using KL-divergence and mRMR.
    
    Parameters:
    - hsi_image: 3D numpy array of shape (height, width, bands)
    - labels: 2D numpy array of shape (height, width)
    - num_bands: Number of bands to select

    Returns:
    - selected_bands: List of selected band indices
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    data = hsi_image.reshape(n_samples, n_bands)
    labels_flat = labels.reshape(-1)
    
    # KL divergence between each band and the whole distribution
    kl_scores = []
    for i in range(n_bands):
        p = np.histogram(data[:, i], bins=20, density=True)[0]
        q = np.histogram(data, bins=20, density=True)[0]
        kl_scores.append(kl_divergence(p, q))
    
    # Sort by KL divergence
    kl_sorted_bands = np.argsort(kl_scores)
    
    # Select the top KL-scored bands
    selected_bands = kl_sorted_bands[:num_bands]
    data_selected = hsi_image[:, :, selected_bands].reshape(n_samples, num_bands)
    
    # Create DataFrame for mRMR
    df = pd.DataFrame(data_selected)
    df['label'] = labels_flat.astype(str)
    
    # Discretize continuous data for mRMR
    for col in df.columns[:-1]:
        df[col] = pd.qcut(df[col], q=20, duplicates='drop').astype('category').cat.codes
    
    # mRMR feature selection
    selected_features = pymrmr.mRMR(df, 'MIQ', num_bands)
    
    # Convert mRMR selected features to the final selected bands
    final_selected_bands = [selected_bands[int(feature)] for feature in selected_features]

    print(f"Selected Bands after mRMR: {final_selected_bands}")
    
    return final_selected_bands


# Function for model training and grid search CatBoost optimization
def model_train_test(hsi_image, gt, test_size=0.2, random_state=42):
    selected_bands = band_selection_kl_mrmr(hsi_image, gt, num_bands=120)
    hsi_selected = hsi_image[:, :, selected_bands]
    
    # Reshape data
    n_samples = hsi_selected.shape[0] * hsi_selected.shape[1]
    X = hsi_selected.reshape(n_samples, -1)
    y = gt.reshape(-1)

    # Remove unlabeled data (assuming 0 is the background/unlabeled class)
    mask = y > 0
    X = X[mask]
    y = y[mask]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    # Dimensionality reduction with PCA (optional)
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # CatBoost with Grid Search for parameter optimization
    catboost = CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', random_seed=random_state)

    param_grid = {
        'iterations': [500, 1000],
        'depth': [4, 6],
        'learning_rate': [0.03, 0.1],
        'l2_leaf_reg': [3, 5],
    }

    grid_search = GridSearchCV(catboost, param_grid, cv=3, scoring='accuracy', verbose=3)
    
    start = time.time()
    grid_search.fit(X_train_pca, y_train)
    end = time.time()

    best_catboost = grid_search.best_estimator_
    
    # Prediction and evaluation
    y_pred = best_catboost.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    # Classification report
    print(f"Best parameters: {grid_search.best_params_}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Reshape for full classification map
    hsi_image_reshaped = hsi_selected.reshape(-1, hsi_selected.shape[2])
    hsi_image_pca = pca.transform(hsi_image_reshaped)
    y_pred_full = best_catboost.predict(hsi_image_pca)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])

    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])
    
    return acc, end - start, classification_map, ground_truth_map


# Visualization of classification results
def visualize_classification_map(classification_map, ground_truth_map, dataset_name):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(ground_truth_map, cmap='jet')
    plt.title(f"Ground Truth - {dataset_name}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap='jet')
    plt.title(f"Classification Map - {dataset_name}")
    plt.axis('off')

    plt.savefig(f"{dataset_name}_classification_vs_ground_truth.png")
    print(f"Figure saved as {dataset_name}_classification_vs_ground_truth.png")


# Load datasets (replace file paths with actual dataset locations)
pavia_u = scipy.io.loadmat('PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('PaviaU_gt.mat')['paviaU_gt']

pavia_c = scipy.io.loadmat('Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('Pavia_gt.mat')['pavia_gt']

salinas = scipy.io.loadmat('Salinas.mat')['salinas']
salinas_gt = scipy.io.loadmat('Salinas_gt.mat')['salinas_gt']

indian_pines = scipy.io.loadmat('Indian_pines.mat')['indian_pines']
indian_pines_gt = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']


# Train, test, and visualize for Pavia University
acc_pavia_u, training_time_pavia_u, classification_map_pavia_u, ground_truth_map_pavia_u = model_train_test(pavia_u, pavia_u_gt)
visualize_classification_map(classification_map_pavia_u, ground_truth_map_pavia_u, "Pavia University")

# Train, test, and visualize for Pavia Centre
acc_pavia_c, training_time_pavia_c, classification_map_pavia_c, ground_truth_map_pavia_c = model_train_test(pavia_c, pavia_c_gt)
visualize_classification_map(classification_map_pavia_c, ground_truth_map_pavia_c, "Pavia Centre")

# Train, test, and visualize for Salinas
acc_salinas, training_time_salinas, classification_map_salinas, ground_truth_map_salinas = model_train_test(salinas, salinas_gt)
visualize_classification_map(classification_map_salinas, ground_truth_map_salinas, "Salinas")

# Train, test, and visualize for Indian Pines
acc_indian_pines, training_time_indian_pines, classification_map_indian_pines, ground_truth_map_indian_pines = model_train_test(indian_pines, indian_pines_gt)
visualize_classification_map(classification_map_indian_pines, ground_truth_map_indian_pines, "Indian Pines")


# Print accuracies and training times
print(f"Pavia University - Training Time: {training_time_pavia_u:.2f} sec, Accuracy: {acc_pavia_u * 100:.2f}%")
print(f"Pavia Centre - Training Time: {training_time_pavia_c:.2f} sec, Accuracy: {acc_pavia_c * 100:.2f}%")
print(f"Salinas - Training Time: {training_time_salinas:.2f} sec, Accuracy: {acc_salinas * 100:.2f}%")
print(f"Indian Pines - Training Time: {training_time_indian_pines:.2f} sec, Accuracy: {acc_indian_pines * 100:.2f}%")

