import numpy as np
import scipy.io
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import time


# @brief: Visualize the classification map and ground truth side by side.
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
    os.makedirs('maps/1.baseline_GPU', exist_ok=True)
    filepath = os.path.join('maps/1.baseline_GPU', f"{dataset_name}_classification_vs_ground_truth.png")
    
    # Save the figure as an image
    plt.savefig(filepath)
    print(f"Figure saved as {filepath}")


# Function to apply SMOTE
def apply_smote(X_train, y_train, random_state=42):
    """
    This function applies SMOTE to the training data to balance the class distribution.
    """
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


# Dataset classification with Grid Search
def initial_model(hsi_image, gt, test_size=0.2, random_state=42):
    """
    Method to create and train a CatBoost model using hyperspectral imagery data.
    The method will also apply SMOTE for balancing classes, test the model, and generate a classification map.
    """
    
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.1],  # Example range
        'depth': [3, 4, 6],                  # Tree depth
        'n_estimators': [200, 400, 1000],    # Number of boosting iterations
        'l2_leaf_reg': [1, 3, 5],            # L2 regularization
        'loss_function': ['MultiClass']      # Loss function for classification
    }

    # Reshape hyperspectral image data
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reshaped, gt.reshape(-1), 
                                                        stratify=gt.reshape(-1), test_size=test_size, random_state=42)

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Initialize CatBoost model
    cbc = CatBoostClassifier(task_type='GPU', verbose=False)

    # Set up Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=cbc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)

    # Step 4: Train the model with Grid Search
    start = time.time()
    grid_search.fit(X_train_smote, y_train_smote)
    end = time.time()
    train_time = end - start

    # Print the best hyperparameters and all the results from grid search
    print("Best Hyperparameters found by GridSearchCV:", grid_search.best_params_)
    print(f"Training Time: {train_time} seconds")
    
    # Get all the results from grid search (to see how other hyperparameter combinations performed)
    cv_results = grid_search.cv_results_
    
    # Show all hyperparameter combinations and their corresponding mean test scores
    print("\nGrid Search Results for all combinations:")
    for i in range(len(cv_results['params'])):
        print(f"Combination {i+1}: {cv_results['params'][i]}, Mean Test Score: {cv_results['mean_test_score'][i]}")

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Test the model on the test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {acc}")

    # Generate the classification map for the entire hyperspectral dataset using the best model
    y_pred_full = best_model.predict(hsi_image_reshaped)

    # Reshape the predicted labels back into the original image shape (height x width)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Reshape the ground truth labels to the original shape (height x width)
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Return accuracy, training time, classification map, and ground truth map
    return acc, train_time, classification_map, ground_truth_map


# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']

# Train and evaluate the model
acc, train_time, classification_map, ground_truth_map = initial_model(pavia_u, pavia_u_gt)

# Visualize the classification map
visualize_classification_map(classification_map, ground_truth_map, "Pavia_university_GS")

