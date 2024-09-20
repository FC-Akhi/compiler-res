"""
This is the implementation of 
- CatBoost(Parameters taken by GS) with SMOTE data balancing technique, 
- Catboost Feature scores based band reduction,
- Genetic Algorithm for band combination and having best band combination.

Datasets:
- Pavia University (Pavia U)
- Pavia Centre (Pavia C)


Architecture:
- GPU-based implementation using CatBoost for classification.

Program Description:


Authors:
- FC-Akhi, Nikhil Badoni
- Date: 14th Sept. 2024
"""
import pickle
# Import necessary libraries
import os  # For directory operations
import numpy as np  # For numerical computations and array manipulations
import time  # To calculate training time
import random # importing random
import math

import scipy  # For scientific computations and loading datasets
from catboost import CatBoostClassifier  # For the CatBoost classifier model
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split  # To split dataset into training and test sets
import matplotlib.pyplot as plt  # For plotting graphs and visualization
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets by oversampling
from collections import Counter  # To count occurrences of each class label
# importing the 'accuracy_score' function from the 'sklearn' library for evaluating classification accuracy
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import median_filter
from collections import Counter



# population size
no_of_population = 20

# percent of bands
percent_of_bands = 80


# generation size
no_of_generations = 10



# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']


# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']


# Assuming this is your function for metrics calculation
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
def initial_model(hsi_image, gt, test_size=0.2, random_state=42):
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
    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reshaped, 
                                                        gt.reshape(-1), 
                                                        stratify=gt.reshape(-1), 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    

    # Debugging: Check class distribution before SMOTE
    print("Class distribution before SMOTE (Training set):", Counter(y_train))

    # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Debugging: Check class distribution after SMOTE
    print("Class distribution after SMOTE (Training set):", Counter(y_train_smote))

    

    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        n_estimators=1500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_strength=2,
        l2_leaf_reg=1,
        task_type='GPU',
        early_stopping_rounds=None,
        verbose=500
    )


    # Train the CatBoost model on the SMOTE-applied training data
    cbc.fit(X_train_smote, y_train_smote)

    # Debugging: Check the unique classes in the test data and training data
    print("Unique classes in training set after SMOTE:", np.unique(y_train_smote))
    print("Unique classes in test set:", np.unique(y_test))


    # return the trained model
    return cbc


def find_feature_score(hsi_image, gt):
    """
    Method for extracting feature scores for the spectral bands in a hyperspectral image.
    
    Input:
    - hsi_image: 3D array of hyperspectral image data (height x width x bands)
    - gt: Ground truth labels (height x width)
    
    Output:
    - feature_score: Feature importance scores of each band in the hyperspectral image
    """

    # Train the model and get the CatBoost model instance
    cbc = initial_model(hsi_image, gt)

    # Get feature importances using the 'PredictionValuesChange' method
    feature_score = cbc.get_feature_importance(type='PredictionValuesChange')

    print("Feature score generation done")

    # Return the feature score
    return feature_score


# Normalize the feature scores to a scale between 0 and 1
def normalize_scores(scores):
    """
    Normalize the feature scores to a range between 0 and 1.
    
    Input:
    - scores: Array of feature scores
    
    Output:
    - Normalized feature scores
    """
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


def get_reduced_bands(feature_score_list):
    """
    Function to reduce the number of bands by selecting those with the highest feature scores.
    
    Input:
    - feature_score_list: Array of feature importance scores
    
    Output:
    - new_sorted_score_list: Sorted list of high-ranking scores
    - new_indices_list: Indices of the high-ranking bands
    - deleted_indices_list: Indices of the bands that are deleted
    """

    new_sorted_score_list = []
    new_indices_list = []
    deleted_indices_list = []

    # Sort the feature scores in descending order (best scores first)
    sorted_list_scores = np.sort(feature_score_list)[::-1]
    sorted_indices = np.argsort(feature_score_list)[::-1]

    # Calculate the median score to use as a threshold for band selection
    median = np.median(sorted_list_scores)

    print("Median score:", median)

    # Select bands with scores greater than or equal to the median score
    for i in range(len(sorted_list_scores)):
        score = sorted_list_scores[i]
        index = sorted_indices[i]
        if score >= median:
            new_sorted_score_list.append(score)
            new_indices_list.append(index)
        else:
            deleted_indices_list.append(index)

    # Return sorted score list, selected band indices, and deleted band indices
    return new_sorted_score_list, new_indices_list, deleted_indices_list


def band_removal_and_model_train(hsi_image, gt, deleted_indices, test_size=0.2, random_state=42):
    """
    Function to train a model after removing certain spectral bands.
    
    Input:
    - hsi_image: 3D array of hyperspectral image data (height x width x bands)
    - gt: Ground truth labels (height x width)
    - deleted_indices: List of band indices to delete from the dataset
    - test_size: Proportion of the data to be used for testing (default = 0.2)
    - random_state: Random seed for reproducibility (default = 42)
    
    Output:
    - acc: Accuracy score of the model on the test data
    - test_time: Time taken to train the model
    - classification_map: Predicted class labels for the entire hyperspectral image
    - ground_truth_map: Ground truth labels
    """

    # Reshape the 3D hyperspectral image into a 2D array (samples x bands)
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Remove the bands corresponding to deleted indices
    hsi_image_reduced = np.delete(hsi_image_reshaped, deleted_indices, axis=1)

    # Split the data into training and testing sets, stratified by the ground truth labels
    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reduced, 
                                                        gt.reshape(-1), 
                                                        stratify=gt.reshape(-1), 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    # Debugging: Check class distribution before SMOTE after band removal
    print("Class distribution before SMOTE (Training set after band removal):", Counter(y_train))
    
    # Apply SMOTE to balance the training data
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Debugging: Check class distribution after SMOTE
    print("Class distribution after SMOTE (Training set after band removal):", Counter(y_train_smote))


    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        n_estimators=1500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_strength=2,
        l2_leaf_reg=1,
        task_type='GPU',
        early_stopping_rounds=None,
        verbose=500
    )

    # Train the model on the SMOTE-balanced training data
    
    cbc.fit(X_train_smote, y_train_smote)
    
    print("Training completed successfully.")

    # Debugging: Check the unique classes in the test data
    # print("Unique classes in test set:", np.unique(y_test))


    # Test the model on the test set
    start = time.time()
    y_pred = cbc.predict(X_test)
    end = time.time()
    test_time = end - start  # Calculate the training time

    print("Unique classes predicted:", np.unique(y_pred))

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)


    # Generate the classification map for the entire hyperspectral dataset
    y_pred_full = cbc.predict(hsi_image_reduced)

    # Reshape the predicted labels back into the original image shape (height x width)
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Reshape the ground truth labels to the original shape (height x width)
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])

    # Return accuracy, training time, classification map, and ground truth map
    return oa, test_time, classification_map, ground_truth_map

# ===============================++++++++++++++++++++++++==========================================

def get_size_of_single_combination(band_numbers, percent_of_bands):

    # 60% of bands are taken from the reduced band set came from step 3
    no_of_bands = len(band_numbers) * (percent_of_bands / 100)


    return math.ceil(no_of_bands)

def blackbox_function(band_combination, hsi_image, gt, feature_importance, test_size = 0.2, random_state=42):

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

    # empty list for deleted_indices
    deleted_indices = []

    # listing out the indices as deleted_indices which are not in band combination
    for index, num in enumerate(feature_importance):
        if index in band_combination:
            continue
        else:
            deleted_indices.append(index)

    

    hsi_image_reduced = np.delete(hsi_image_reshaped, deleted_indices, axis=1)  # remove the deleted bands from the HSI image


    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reduced, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)  # split the data into training and testing sets
    
    # Create CatBoost classifier
    cbc = CatBoostClassifier(
        n_estimators=1500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        random_strength=2,
        l2_leaf_reg=1,
        task_type='GPU',
        early_stopping_rounds=None,
        verbose=500  
    )

    

    
    # Train the model
    start_train_time = time.time()
    cbc.fit(X_train, y_train)  # train the classifier on the training set
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    print("Training completed successfully.")
    # Debugging: Check the unique classes in the test data
    # print("Unique classes in test set:", np.unique(y_test))


    # Test the model
    start_test_time = time.time()
    y_pred = cbc.predict(X_test)
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time

    # print("Unique classes predicted:", np.unique(y_pred))

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)



    # Generate classification map for the entire dataset
    y_pred_full = cbc.predict(hsi_image_reduced)
    
    
    classification_map = y_pred_full.reshape(hsi_image.shape[0], hsi_image.shape[1])
    
    # # Reshape ground truth
    ground_truth_map = gt.reshape(hsi_image.shape[0], hsi_image.shape[1])


    # Return accuracy, training time, and classification maps
    return oa, testing_time, classification_map, ground_truth_map, cbc


def genetic_algorithm(num_solutions, n_bands_per_solution, threshold_accuracy, num_generations, hsi_image, gt, BANDS_score, BANDS_index, original_feature_score):
    """
    Genetic Algorithm with dynamic crossover and mutation rates, elitism, and probabilistic selection.
    """
    
    def dynamic_mutation_rate(gen, max_gen, base_rate=0.5):
        """ Increase mutation rate early on, then decrease as we approach max generations """
        return base_rate * (1 - (gen / max_gen))  # Higher mutation rate early, reduces over time

    def dynamic_crossover_rate(gen, max_gen, base_rate=0.7):
        """ Reduce crossover rate as generations progress for more exploration early, refinement later """
        return base_rate * (gen / max_gen)

    # classification map file counter
    c = 1

    # empty list for initial population
    population = []

    # INITIALIZATION: Generate the initial population
    for i in range(num_solutions):
        band_indices = []

        # Select n bands for a population
        while len(band_indices) < n_bands_per_solution:
            idx = random.choice(range(len(BANDS_index)))
            if idx not in band_indices:
                band_indices.append(idx)

        band_combination = [BANDS_index[idx] for idx in band_indices]

        # EVALUATION: Evaluate the accuracy of the n-band combination using the objective function
        accuracy, testing_time, classification_map, ground_truth_map, cbc = blackbox_function(band_combination, hsi_image, gt, original_feature_score)

        # SELECTION: Keep the n-band combination if its accuracy is greater than the threshold accuracy
        if accuracy > threshold_accuracy:
            population.append((band_combination, accuracy, test_time))
            # print("parent solution:", band_combination)
            # print("parents accuracy", accuracy)
            # print("parents test_time", test_time)

    # ELITISM: Ensure best individual is always kept
    elite_size = max(1, int(0.2 * num_solutions))  # Top 10% of population kept as elite

    # Evolve the population over generations
    for gen in range(num_generations):
        new_population = []

        # Sort population by accuracy (descending order)
        population.sort(key=lambda x: x[1], reverse=True)
        
        # Carry over the elite individuals
        new_population.extend(population[:elite_size])

        # Adjust mutation and crossover rates dynamically
        mutation_rate = dynamic_mutation_rate(gen, num_generations)
        crossover_rate = dynamic_crossover_rate(gen, num_generations)

        # Perform crossover & mutation to generate new offspring
        for i in range(num_solutions - elite_size):  # Generate offspring for non-elite portion of population

            # Probabilistic parent selection (Roulette wheel based on accuracy)
            parent1 = random.choices(population, weights=[p[1] for p in population])[0][0]
            parent2 = random.choices(population, weights=[p[1] for p in population])[0][0]

            # RECOMBINATION / CROSSOVER
            offspring = []
            j = 0
            while len(offspring) < n_bands_per_solution:
                if random.random() < crossover_rate and parent1[j % len(parent1)] not in offspring:
                    offspring.append(parent1[j % len(parent1)])
                elif random.random() >= crossover_rate and parent2[j % len(parent2)] not in offspring:
                    offspring.append(parent2[j % len(parent2)])
                j += 1

            # MUTATION: Mutate a single band in the offspring solution
            for j in range(n_bands_per_solution):
                if random.random() < mutation_rate and len(offspring) > 0:
                    index = random.randint(0, len(offspring) - 1)
                    new_band = random.choice([b for b in BANDS_index if b not in offspring])
                    offspring[index] = new_band

            # EVALUATION: Evaluate the accuracy of the offspring solution
            accuracy, testing_time, classification_map, ground_truth_map, cbc = blackbox_function(offspring, hsi_image, gt, original_feature_score)

            # SELECTION: Keep the offspring solution if its accuracy is greater than the threshold accuracy
            if accuracy > threshold_accuracy:
                new_population.append((offspring, accuracy, test_time))
                # print("offspring combination:", offspring)
                # print("offspring accuracy", accuracy)
                # print("offspring test_time", test_time)

        # Combine elite and new offspring
        population = population[:elite_size] + new_population

        # Sort combined population by accuracy again
        population.sort(key=lambda x: x[1], reverse=True)

    # Return the band combinations with accuracy
    return population, cbc




# Pavia University
# +++++++++++++======+++++++++==========++++++++++++++++++++++
# Find feature score Pavia University
feature_score_pavia_u = find_feature_score(pavia_u, pavia_u_gt)

# Normalized CatBoost feature scores
normalized_cbc_score_pavia_u = normalize_scores(feature_score_pavia_u)

# Reduce bands depending on feature score
new_sorted_score_list_pavia_u, new_indices_list_pavia_u, deleted_indices_list_pavia_u = get_reduced_bands(normalized_cbc_score_pavia_u)

# Train on reduced data set and find the accuracy which will be the threshold accuracy for GA
threshold_acc_pavia_u, test_time, classification_map, ground_truth_map = band_removal_and_model_train(pavia_u, pavia_u_gt, deleted_indices_list_pavia_u)

# # candidate solution size per population
no_of_bands_per_solution = get_size_of_single_combination(new_indices_list_pavia_u, percent_of_bands)

# list of different band combinations equa to or higher than threshold accuracy
solutions_pavia_u, cbc_u = genetic_algorithm(
    no_of_population, 
    no_of_bands_per_solution, 
    threshold_acc_pavia_u, 
    no_of_generations, 
    pavia_u, 
    pavia_u_gt, 
    new_sorted_score_list_pavia_u, 
    new_indices_list_pavia_u, 
    normalized_cbc_score_pavia_u
)


# Directly access the best solution
best_solution = solutions_pavia_u[0]

# Unpack the tuple from best_solution (contains band combination, accuracy, test time)
best_band_combination_u = best_solution[0]  # This is the list of bands
accuracy = best_solution[1]               # This is the accuracy value
test_time = best_solution[2]              # This is the test time value

# Print out the details
print("Best band combination:", best_band_combination_u)
print("Number of bands:", len(best_band_combination_u))
print(f"PU - Accuracy: {accuracy * 100:.2f}%")
print(f"PU - time_train: {test_time} seconds")

# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_u.save_model('catboost_pavia_u_model.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('reduced_band_combination_u.pkl', 'wb') as f:
    pickle.dump(best_band_combination_u, f)




# Pavia centre
# +++++++++++++++++++++++++++++++++++========+++++++++++++++++++++++++

# Find feature score Pavia University
feature_score_pavia_c = find_feature_score(pavia_c, pavia_c_gt)

# Normalized CatBoost feature scores
normalized_cbc_score_pavia_c = normalize_scores(feature_score_pavia_c)

# Reduce bands depending on feature score
new_sorted_score_list_pavia_c, new_indices_list_pavia_c, deleted_indices_list_pavia_c = get_reduced_bands(normalized_cbc_score_pavia_c)

# Train on reduced data set and find the accuracy which will be the threshold accuracy for GA
threshold_acc_pavia_c, test_time, classification_map, ground_truth_map = band_removal_and_model_train(pavia_c, pavia_c_gt, deleted_indices_list_pavia_c)

# # candidate solution size per population
no_of_bands_per_solution = get_size_of_single_combination(new_indices_list_pavia_c, percent_of_bands)

# list of different band combinations equa to or higher than threshold accuracy
solutions_pavia_c, cbc_c = genetic_algorithm(
    no_of_population, 
    no_of_bands_per_solution, 
    threshold_acc_pavia_c, 
    no_of_generations, 
    pavia_c, 
    pavia_c_gt, 
    new_sorted_score_list_pavia_c, 
    new_indices_list_pavia_c, 
    normalized_cbc_score_pavia_c
)


# Directly access the best solution
best_solution = solutions_pavia_c[0]

# Unpack the tuple from best_solution (contains band combination, accuracy, test time)
best_band_combination_c = best_solution[0]  # This is the list of bands
accuracy = best_solution[1]               # This is the accuracy value
test_time = best_solution[2]              # This is the test time value

# Print out the details
print("Best band combination:", best_band_combination_c)
print("Number of bands:", len(best_band_combination_c))
print(f"PC - Accuracy: {accuracy * 100:.2f}%")
print(f"PC - time_train: {test_time} seconds")

# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_c.save_model('catboost_pavia_c_model.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('reduced_band_combination_c.pkl', 'wb') as f:
    pickle.dump(best_band_combination_c, f)
