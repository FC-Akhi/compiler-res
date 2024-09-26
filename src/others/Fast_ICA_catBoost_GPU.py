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
import random

import math

# population size
no_of_population = 40

# percent of bands
percent_of_bands = 50


# generation size
no_of_generations = 10




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






# Blackbox function to evaluate band combination accuracy using CatBoost
def blackbox_function(band_combination, hsi_image, gt, test_size=0.2, random_state=42):
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Remove bands not in the combination
    deleted_indices = [i for i in range(n_bands) if i not in band_combination]
    hsi_image_reduced = np.delete(hsi_image_reshaped, deleted_indices, axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(hsi_image_reduced, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=random_state)

    # Train CatBoost model with early stopping
    cbc = CatBoostClassifier(
        n_estimators=1500, 
        learning_rate=0.1, 
        depth=6, 
        loss_function='MultiClass', 
        task_type='GPU', 
        early_stopping_rounds=10,  # Early stopping to avoid long training times
        verbose=False
    )

    start_train_time = time.time()
    cbc.fit(X_train_smote, y_train_smote, eval_set=(X_test, y_test))
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Test the model
    y_pred = cbc.predict(X_test)
    testing_time = time.time() - end_train_time

    # Calculate metrics: OA, AA, and Kappa
    oa, aa, kappa = calculate_metrics(y_test, y_pred)

    return oa, testing_time


def genetic_algorithm(num_solutions, percent_of_bands, num_generations, hsi_image, gt):
    """
    Genetic Algorithm for hyperspectral band selection.
    """
    def dynamic_mutation_rate(gen, max_gen, base_rate=0.5):
        return base_rate * (1 - (gen / max_gen))

    def dynamic_crossover_rate(gen, max_gen, base_rate=0.7):
        return base_rate * (gen / max_gen)

    n_bands_total = hsi_image.shape[2]
    no_of_bands_per_solution = math.ceil(n_bands_total * (percent_of_bands / 100))

    print(f"Each band combination will have {no_of_bands_per_solution} bands.")

    # Population to store solutions
    population = []

    # Initial band combination evaluation to set threshold accuracy
    first_combination = random.sample(range(n_bands_total), no_of_bands_per_solution)
    threshold_accuracy, testing_time = blackbox_function(first_combination, hsi_image, gt)
    print(f"Initial threshold accuracy: {threshold_accuracy:.2f}")

    # Ensure that at least the first combination is added to the population
    population.append((first_combination, threshold_accuracy, testing_time))

    # Generate the rest of the population
    for i in range(num_solutions):
        band_combination = random.sample(range(n_bands_total), no_of_bands_per_solution)
        accuracy, testing_time = blackbox_function(band_combination, hsi_image, gt)

        if accuracy > threshold_accuracy:
            population.append((band_combination, accuracy, testing_time))

    elite_size = max(1, int(0.2 * num_solutions))

    # Evolve the population through generations
    for gen in range(num_generations):
        new_population = []

        # Sort population by accuracy in descending order
        population.sort(key=lambda x: x[1], reverse=True)

        # Keep the elite individuals
        new_population.extend(population[:elite_size])

        mutation_rate = dynamic_mutation_rate(gen, num_generations)
        crossover_rate = dynamic_crossover_rate(gen, num_generations)

        # Crossover and mutation to generate offspring
        for i in range(num_solutions - elite_size):
            parent1 = random.choice(population)[0]
            parent2 = random.choice(population)[0]

            # Crossover: Combine parts of parent1 and parent2
            offspring = list(set(parent1[:no_of_bands_per_solution // 2] + parent2[no_of_bands_per_solution // 2:]))

            # If offspring length is less than the required size, add random bands
            while len(offspring) < no_of_bands_per_solution:
                random_band = random.choice(range(n_bands_total))
                if random_band not in offspring:
                    offspring.append(random_band)

            # If offspring length exceeds the required size, trim it down
            if len(offspring) > no_of_bands_per_solution:
                offspring = offspring[:no_of_bands_per_solution]

            # Mutation: Randomly replace some elements in the offspring
            if random.random() < mutation_rate:
                index_to_mutate = random.randint(0, no_of_bands_per_solution - 1)
                offspring[index_to_mutate] = random.choice([b for b in range(n_bands_total) if b not in offspring])

            # Evaluate the offspring
            accuracy, testing_time = blackbox_function(offspring, hsi_image, gt)

            if accuracy > threshold_accuracy:
                new_population.append((offspring, accuracy, testing_time))

        # Combine elite and new offspring
        population = new_population

    # Return the final population
    return population







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
        verbose=False)
    
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



# Function to apply ICA for dimensionality reduction on a reduced set of bands
def ICA(hsi_image, gt, best_band_combination):
    """
    Apply ICA for dimensionality reduction on a reduced set of bands provided by the GA.
    
    Args:
        hsi_image (ndarray): The hyperspectral image (height x width x bands).
        gt (ndarray): Ground truth labels (height x width).
        best_band_combination (list): The best band combination from the Genetic Algorithm.
        
    Returns:
        oa (float): Overall Accuracy.
        aa (float): Average Accuracy.
        kappa (float): Kappa score.
        total_time1 (float): Training time.
        total_time2 (float): Testing time.
        cbc (CatBoostClassifier): Trained CatBoost classifier.
        hsi_image_limited (ndarray): Reduced hyperspectral image based on the best band combination.
    """
    n_samples = hsi_image.shape[0] * hsi_image.shape[1]
    n_bands = hsi_image.shape[2]
    
    # Reshape the hyperspectral image into 2D array (samples x bands)
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)

    # Apply the best band combination to reduce the hyperspectral image
    hsi_image_reduced = hsi_image_reshaped[:, best_band_combination]

    # Standardize the data before applying ICA
    scaler = StandardScaler()
    hsi_image_scaled = scaler.fit_transform(hsi_image_reduced)

    # Apply ICA to the reduced band set
    ica = FastICA(random_state=42)
    hsi_image_limited = ica.fit_transform(hsi_image_scaled)

    # Print the number of bands before and after applying ICA
    print(f"Number of bands before applying ICA: {len(best_band_combination)}")
    print(f"Number of components after applying ICA: {hsi_image_limited.shape[1]}")  # Print remaining components

    # Train the model on the reduced set of bands after ICA
    oa, aa, kappa, total_time1, total_time2, cbc = model_train_test(hsi_image, gt, hsi_image_limited)

    return oa, aa, kappa, total_time1, total_time2, cbc, hsi_image_limited









solutions_pavia_u = genetic_algorithm(no_of_population, percent_of_bands, no_of_generations, pavia_u, pavia_u_gt)


# Directly access the best solution
best_solution_u = solutions_pavia_u[0]

# Unpack the tuple from best_solution_u (contains band combination, accuracy, test time)
best_band_combination_u = best_solution_u[0]  # This is the list of bands
accuracy = best_solution_u[1]               # This is the accuracy value
test_time = best_solution_u[2]              # This is the test time value

# Print out the details
print("Best band combination:", best_band_combination_u)
print("Number of bands:", len(best_band_combination_u))
print(f"PU - Accuracy: {accuracy * 100:.2f}%")
print(f"PU - time_train: {test_time} seconds")




# # Train and test using ICA
oa_u, aa_u, kappa_u, train_time, testing_time_u, cbc_u, hsi_image_ica_transformed_u = ICA(pavia_u, pavia_u_gt, best_band_combination_u)


# Print the accuracy metrics
# Print metrics for Pavia University
print(f"Pavia University - Overall Accuracy: {oa_u * 100:.2f}%")
print(f"Pavia University - Average Accuracy: {aa_u* 100:.2f}%")
print(f"Pavia University - Kappa Coefficient: {kappa_u:.4f}")
print(f"Pavia University - Testing Time: {testing_time_u:.2f} sec")



# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_u.save_model('CB-Fast_ICA_model_U.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('CB-Fast_ICA_reduced_band_combination_u.pkl', 'wb') as f:
    pickle.dump(hsi_image_ica_transformed_u, f)










solutions_pavia_c = genetic_algorithm(no_of_population, percent_of_bands, no_of_generations, pavia_c, pavia_c_gt)


# Directly access the best solution
best_solution_c = solutions_pavia_c[0]

# Unpack the tuple from best_solution_c (contains band combination, accuracy, test time)
best_band_combination_c = best_solution_c[0]  # This is the list of bands
accuracy = best_solution_c[1]               # This is the accuracy value
test_time = best_solution_c[2]              # This is the test time value

# Print out the details
print("Best band combination:", best_band_combination_c)
print("Number of bands:", len(best_band_combination_c))
print(f"PC - Accuracy: {accuracy * 100:.2f}%")
print(f"PC - time_train: {test_time} seconds")




# # Train and test using ICA
oa_c, aa_c, kappa_c, train_time, testing_time_c, cbc_c, hsi_image_ica_transformed_c = ICA(pavia_c, pavia_c_gt, best_band_combination_c)


# Print the accuracy metrics
# Print metrics for Pavia University
print(f"Pavia Centre- Overall Accuracy: {oa_c * 100:.2f}%")
print(f"Pavia Centre- Average Accuracy: {aa_c* 100:.2f}%")
print(f"Pavia Centre- Kappa Coefficient: {kappa_c:.4f}")
print(f"Pavia Centre- Testing Time: {testing_time_c:.2f} sec")



# After training the model and obtaining the best solution
# Save the trained CatBoost model
cbc_c.save_model('CB-Fast_ICA_model_c.cbm')  # Saving the CatBoost model in a .cbm file

# Save the reduced set of bands
with open('CB-Fast_ICA_reduced_band_combination_c.pkl', 'wb') as f:
    pickle.dump(hsi_image_ica_transformed_c, f)









