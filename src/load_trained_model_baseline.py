import pickle
import numpy as np
from catboost import CatBoostClassifier
import scipy  # For scientific computations and loading datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import numpy as np
import matplotlib.colors as mcolors




# Load dataset-Pavia university
pavia_u = scipy.io.loadmat('contents/data/PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat('contents/data/PaviaU_gt.mat')['paviaU_gt']


# Load dataset-Pavia centre
pavia_c = scipy.io.loadmat('contents/data/Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat('contents/data/Pavia_gt.mat')['pavia_gt']


# Load the CatBoost model for pavia university
cbc_u_loaded = CatBoostClassifier()
cbc_u_loaded.load_model('src_baseline/catboost_pavia_u_model_baseline.cbm')




# Load the CatBoost model for pavia centre
cbc_c_loaded = CatBoostClassifier()
cbc_c_loaded.load_model('src_baseline/catboost_pavia_c_model_baseline.cbm')



# Create the directory if it doesn't exist
output_dir = 'src_baseline/maps'
os.makedirs(output_dir, exist_ok=True)



# Initial checkings
def check_model_classes(cbc_model, hsi_image, expected_num_classes):
    """
    Check if the CatBoost model is trained with the same number of classes as expected.
    
    Args:
        cbc_model (CatBoostClassifier): The trained CatBoost model.
        hsi_image (ndarray): The hyperspectral image (3D array).
        expected_num_classes (int): The expected number of classes, including the background.
    
    Returns:
        bool: True if the number of predicted classes matches the expected number, False otherwise.
    """
    h, w, n_bands = hsi_image.shape
    hsi_image_reshaped = hsi_image.reshape(h * w, n_bands)
    
    # Predict the probability output to check the number of classes
    y_proba_full = cbc_model.predict_proba(hsi_image_reshaped)
    
    num_predicted_classes = y_proba_full.shape[1]  # The second dimension represents the number of classes
    print(f"Model predicted {num_predicted_classes} classes. Expected {expected_num_classes} classes.")
    
    return num_predicted_classes == expected_num_classes


# Initial checkings
def check_ground_truth_background(ground_truth):
    """
    Check if the ground truth contains class 0, representing the background.
    
    Args:
        ground_truth (ndarray): The ground truth label array.
    
    Returns:
        bool: True if class 0 is present, False otherwise.
    """
    unique_classes = np.unique(ground_truth)
    print(f"Unique classes in the ground truth: {unique_classes}")
    
    if 0 in unique_classes:
        print("Class 0 (background) is present in the ground truth.")
        return True
    else:
        print("Class 0 (background) is not found in the ground truth.")
        return False




def visualize_and_save_ground_truth_as_rgb(ground_truth, rgb_color_matrix, output_path, title='Ground Truth RGB Image'):
    """
    Visualize and save the ground truth image with class labels mapped to RGB colors.

    Args:
        ground_truth (ndarray): 2D array containing class labels (e.g., 0 to 9).
        rgb_color_matrix (list or ndarray): List or array of RGB color values for each class.
        output_path (str): File path where the visualized image will be saved.
        title (str): Title for the visualization (optional).
    """
    # Get the height and width of the ground truth map
    h, w = ground_truth.shape

    # Create an empty RGB image with the same height and width
    ground_truth_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Map the ground truth class labels to the RGB color matrix
    for class_idx in range(len(rgb_color_matrix)):
        mask = (ground_truth == class_idx)  # Find where the class label is class_idx
        ground_truth_rgb[mask] = rgb_color_matrix[class_idx]  # Assign corresponding RGB color

    # Visualize the ground truth image
    plt.figure(figsize=(6, 6))
    plt.imshow(ground_truth_rgb)
    plt.title(title)
    plt.axis('off')  # Hide the axes

    # Save the ground truth image as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to prevent display in notebooks

    print(f"Ground truth image saved at {output_path}")


def visualize_and_save_prob_map_as_rgb(prob_map, rgb_color_matrix, output_path, title='RGB Image'):
    """
    Convert a 3D probability map to an RGB image and save it.

    Args:
        prob_map (ndarray): 3D array containing probabilities for each class (height x width x num_classes).
        rgb_color_matrix (list or ndarray): List or array of RGB color values for each class.
        output_path (str): File path where the visualized image will be saved.
        title (str): Title for the visualization (optional).
    """
    # Get the height, width, and number of classes from the probability map
    h, w, num_classes = prob_map.shape

    # Flatten the 3D probability map to a 2D array (pixels x classes)
    flattened_prob_map = prob_map.reshape(h * w, num_classes)

    # Inspect flattened probability map before applying colormap
    print(f"Shape of flattened prob map: {flattened_prob_map.shape}")

    # Convert the flattened probability map to RGB pixels
    rgb_pixels = np.dot(flattened_prob_map, rgb_color_matrix)

    # Reshape the RGB pixel array back to the original image shape
    rgb_image = rgb_pixels.reshape(h, w, 3)

    # Convert to unsigned 8-bit integers (for displaying the image)
    rgb_image_uint8 = rgb_image.astype(np.uint8)

    # Visualize the RGB image
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image_uint8)
    plt.title(title)
    plt.axis('off')  # Hide the axes

    # Save the RGB image as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to prevent display in notebooks

    print(f"Image saved at {output_path}")



def visualize_individual_prob_maps(probability_map, output_dir, dataset_name, colormap=None):
    """
    Visualize and save individual probability maps for each class.
    
    Args:
        probability_map (ndarray): 3D array of class probabilities (H, W, num_classes).
        output_dir (str): Directory where the images will be saved.
        dataset_name (str): Name of the dataset.
        colormap (ListedColormap): Custom colormap for the classes (optional).
    """
    os.makedirs(output_dir, exist_ok=True)

    num_classes = probability_map.shape[2]

    for class_idx in range(num_classes):
        # Extract the probability map for this class
        class_prob_map = probability_map[:, :, class_idx]

        # Normalize the probability map for visualization
        class_prob_map_norm = (class_prob_map - class_prob_map.min()) / (class_prob_map.max() - class_prob_map.min())

        # Plot the probability map
        plt.figure(figsize=(5, 5))
        plt.imshow(class_prob_map_norm, cmap=colormap if colormap else 'viridis')
        plt.title(f'{dataset_name} - Class {class_idx} Probability Map')
        plt.axis('off')

        # Save the probability map for this class
        prob_map_filepath = os.path.join(output_dir, f'{dataset_name}_class_{class_idx}_probability_map.png')
        plt.savefig(prob_map_filepath, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Probability map for class {class_idx} saved at {prob_map_filepath}")





def generate_hard_classification_map(prob_map, rgb_color_matrix, output_dir, file_name):
    """
    Generate a hard classification RGB image and save it to a file.
    
    Args:
        prob_map (ndarray): 3D probability map (height x width x num_classes).
        rgb_color_matrix (list or ndarray): RGB colors for each class.
        output_dir (str): Directory path to save the RGB image.
        file_name (str): Name of the output file (with extension, e.g., 'image.png').
    
    Returns:
        classification_rgb_image (ndarray): RGB image for the hard classification map.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the class with the highest probability for each pixel
    class_predictions = np.argmax(prob_map, axis=2)
    
    # Map the class predictions to the corresponding RGB color
    h, w = class_predictions.shape
    classification_rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(len(rgb_color_matrix)):
        mask = (class_predictions == class_idx)
        classification_rgb_image[mask] = rgb_color_matrix[class_idx]
    
    # Create the full file path
    output_path = os.path.join(output_dir, file_name)
    
    # Save the RGB image
    plt.figure(figsize=(6, 6))
    plt.imshow(classification_rgb_image)
    plt.title("Hard Classification Map")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Classification map saved at {output_path}")
    
    return classification_rgb_image



def generate_probability_map(hsi_image, cbc_model, num_classes):
    """
    Generate a pixel-wise probability map using the trained model.
    
    Args:
        hsi_image (ndarray): The hyperspectral image.
        cbc_model (CatBoostClassifier): Trained CatBoost model.
        num_classes (int): The number of classes (in this case, 9).
    
    Returns:
        prob_maps (ndarray): 3D array of probabilities for each class (H, W, num_classes).
    """
    # Reshape HSI image into 2D (N_samples, N_bands)
    h, w, n_bands = hsi_image.shape  # Get height, width, and number of bands
    n_samples = h * w
    
    # Reshape to (N_samples, N_bands)
    hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands)
    
    # Predict probabilities using the trained model
    y_proba_full = cbc_model.predict_proba(hsi_image_reshaped)


    # Checking prediction output
    print(f"Prediction shape: {y_proba_full.shape}")
    print(f"Max probability per pixel: {np.max(y_proba_full, axis=1)}")

    # Slice to keep only the first 9 classes (in case there are 10)
    y_proba_full = y_proba_full[:, :num_classes]

    print(f"Prediction shape: {y_proba_full.shape}")

    # Reshape the probabilities back to (H, W, num_classes)
    prob_maps = np.reshape(y_proba_full, (h, w, num_classes))

    return prob_maps







# Color matrix for Pavia
rgb_color_matrix_u = [
    [0, 0, 0],
    [217, 191, 218],  # Class 1
    [80, 255, 0],     # Class 2
    [87, 255, 255],   # Class 3
    [46, 138, 82],    # Class 4
    [246, 0, 255],    # Class 5
    [248, 164, 4],    # Class 6
    [159, 33, 236],   # Class 7
    [244, 0, 4],      # Class 8
    [253, 255, 0]     # Class 9
]
# For Pavia University
num_classes_u = 10  # You have 9 classes but always add 1 more for background to keep result correct





# Usage example:
is_background_present = check_ground_truth_background(pavia_u_gt)

# Usage example:
is_class_count_correct = check_model_classes(cbc_u_loaded, pavia_u, 10)
if is_class_count_correct:
    print("The CatBoost model is predicting the correct number of classes.")
else:
    print("Mismatch between model's predicted classes and expected number of classes!")

# For Pavia University
visualize_and_save_ground_truth_as_rgb(pavia_u_gt, rgb_color_matrix_u, os.path.join(output_dir, "paviaU_ground_truth.png"), "Pavia University Ground Truth")

prob_map_u = generate_probability_map(pavia_u, cbc_u_loaded, num_classes_u)

visualize_individual_prob_maps(prob_map_u, output_dir+'/prob_maps', 'Pavia_university')

# # Call the function to visualize and save the RGB image
# visualize_and_save_prob_map_as_rgb(prob_map_u, rgb_color_matrix_u, os.path.join(output_dir, "paviaU_clas_map.png"))
generate_hard_classification_map(prob_map_u, rgb_color_matrix_u, output_dir, "PaviaUniversity_class_map")









# For Pavia Centre
rgb_color_matrix_c = [
    [0, 0, 0],
    [31, 0, 254],        # Class 0 (Black for background)
    [36, 129, 0],  # Class 1
    [80, 255, 0],   # Class 7
    [244, 4, 1],   # Class 3
    [142, 71, 5],      # Class 8
    [193, 193, 193],     # Class 2
    [87, 255, 253],    # Class 4
    [246, 112, 0],    # Class 5
    [253, 255, 0],    # Class 6
    
    
]
num_classes_c = 10  # You have 9 classes



# Usage example:
is_class_count_correct = check_model_classes(cbc_c_loaded, pavia_c, 10)
if is_class_count_correct:
    print("The CatBoost model is predicting the correct number of classes.")
else:
    print("Mismatch between model's predicted classes and expected number of classes!")

# For Pavia centre
visualize_and_save_ground_truth_as_rgb(pavia_c_gt, rgb_color_matrix_c, os.path.join(output_dir, "paviaCw_ground_truth.png"), "Pavia centre Ground Truth")

prob_map_c = generate_probability_map(pavia_c, cbc_c_loaded, num_classes_c)

visualize_individual_prob_maps(prob_map_c, output_dir+'/prob_maps', 'Pavia_centre')

# # Call the function to visualize and save the RGB image
# visualize_and_save_prob_map_as_rgb(prob_map_c, rgb_color_matrix_c, os.path.join(output_dir, "paviaU_clas_map.png"))
generate_hard_classification_map(prob_map_c, rgb_color_matrix_c, output_dir, "PaviaCentre_class_map")










