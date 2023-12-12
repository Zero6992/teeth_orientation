import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

class TeethObject:
    def __init__(self, mask):
        self.mask = mask
        self.area = self.calculate_area()
        self.center_coordinate = self.calculate_center_coordinate()
        self.upperOrLowerJaw = None

    def calculate_area(self):
        return cv2.countNonZero(self.mask)

    def calculate_center_coordinate(self):
        # Find contours of the mask
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the center of the largest contour (assuming it's one connected component)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            else:
                return None
        else:
            return None
        
    def plot_with_center(self):
        # Convert the mask to BGR format for visualization
        mask_bgr = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)

        # Plot the center point on the mask
        if self.center_coordinate:
            cv2.circle(mask_bgr, self.center_coordinate, 5, (0, 0, 255), -1)  # Red dot

        # Display the image with the center point
        plt.imshow(mask_bgr)
        plt.title('Teeth Object with Center Point')
        plt.show()


def get_color_mask(img, target_color):
    img_array = np.array(img)

    # Create a mask for non-colored pixels
    mask = np.all(img_array[:, :, :3] != target_color, axis=-1)

    # # Display the mask as a binary image
    # plt.subplot(1, 3, 1)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Mask')

    # Set non-colored pixels to black
    img_array[mask, :3] = [0, 0, 0]

    # Create a new image from the NumPy array
    new_img = Image.fromarray(img_array)

    # # Display the new image
    # plt.subplot(1, 3, 2)
    # plt.imshow(new_img)
    # plt.title('Modified Image')

    # # Display the target color
    # plt.subplot(1, 3, 3)
    # plt.imshow([[target_color]])
    # plt.title('Target Color')

    # plt.show()

    # color = get_unique_colors(np.array(new_img))
    # print(color)

    return mask, new_img


def binarizeImage(image):
    # Convert PIL Image to NumPy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise ValueError("Unsupported input type. Please provide a PIL Image or a NumPy array.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Use cv2.threshold to create a binary image
    ret, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    return binary_image

def get_unique_colors(image):
    # Convert the image to a NumPy array
    img_array = np.array(image)
    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)

    return unique_colors

#Get bounding boxes from mask of one colour
def get_bounding_box(mask, color):
    # get bounding box from mask
    y_indices, x_indices = np.where(mask == color)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


image_path = 'orientation/1/mask/mask_IMG_2899.png'
image = Image.open(image_path)

teethList = []
colors = get_unique_colors(image)

for color in colors:
    black_color = np.array([0, 0, 0])

    # Check if the color is not black
    if not np.array_equal(color, black_color):
        _, new_img = get_color_mask(image, color)

        # Convert new_img to binary image
        binary_image = binarizeImage(new_img)

        teeth_object = TeethObject(binary_image)
        teethList.append(teeth_object)

# plot original image with center points overlaid
plt.imshow(image)
pts = np.empty((0, 2), dtype=int)

# Append coordinates to the array
for teeth in teethList:
    center_coord = teeth.calculate_center_coordinate()
    if center_coord is not None:
        pts = np.append(pts, [center_coord], axis=0)


# Extract y-coordinates (height) from center coordinates
y_coordinates = np.array([teeth_obj.calculate_center_coordinate()[1] for teeth_obj in teethList if teeth_obj.calculate_center_coordinate() is not None]).reshape(-1, 1)

# Fit KMeans with two clusters on y-coordinates only
# kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=600, tol=1e-4, random_state=42)
y_kmeans = kmeans.fit_predict(y_coordinates)

# Assign upperOrLowerJaw attribute based on clustering result
for idx, label in enumerate(y_kmeans):
    teethList[idx].upperOrLowerJaw = label

# Plot original image with red dots at center points
plt.imshow(image)

# Plot teeth based on upperOrLowerJaw attribute
for label in range(2):
    filtered_teeth = [teeth_obj for teeth_obj in teethList if teeth_obj.upperOrLowerJaw == label]
    x_coords = [teeth_obj.calculate_center_coordinate()[0] for teeth_obj in filtered_teeth]
    y_coords = [teeth_obj.calculate_center_coordinate()[1] for teeth_obj in filtered_teeth]
    color = 'red' if label == 0 else 'blue'
    plt.scatter(x_coords, y_coords, color=color, marker='o', s=50, label=f'Jaw {label + 1}')

# Add legend
plt.legend()

plt.title('Original Image with Teeth Center Points and K-Means Clustering on Height')
plt.show()




# # Fit Agglomerative Clustering with two clusters
# agglomerative = AgglomerativeClustering(n_clusters=2, linkage='ward')
# y_agglomerative = agglomerative.fit_predict(y_coordinates)

# # Assign upperOrLowerJaw attribute based on clustering result
# for idx, label in enumerate(y_agglomerative):
#     teethList[idx].upperOrLowerJaw = label

# # Plot original image with red dots at center points
# plt.imshow(image)

# # Plot teeth based on upperOrLowerJaw attribute
# for label in range(2):
#     filtered_teeth = [teeth_obj for teeth_obj in teethList if teeth_obj.upperOrLowerJaw == label]
#     x_coords = [teeth_obj.calculate_center_coordinate()[0] for teeth_obj in filtered_teeth]
#     y_coords = [teeth_obj.calculate_center_coordinate()[1] for teeth_obj in filtered_teeth]
#     color = 'red' if label == 0 else 'blue'
#     plt.scatter(x_coords, y_coords, color=color, marker='o', s=50, label=f'Jaw {label + 1}')

# # Add legend
# plt.legend()

# plt.title('Original Image with Teeth Center Points and Agglomerative Clustering on Height')
# plt.show()


# # Extract x and y coordinates from center coordinates
# coordinates = np.array([teeth_obj.calculate_center_coordinate() for teeth_obj in teethList if teeth_obj.calculate_center_coordinate() is not None])

# # Fit KMeans with six clusters on both x and y coordinates
# kmeans = KMeans(n_clusters=6, init='k-means++', n_init=20, max_iter=600, tol=1e-4, random_state=42)
# cluster_labels = kmeans.fit_predict(coordinates)

# # Assign distinct hues to each cluster label
# num_clusters = len(set(cluster_labels))
# hue_values = np.linspace(0, 1, num_clusters, endpoint=False)

# # Assign upperOrLowerJaw attribute based on clustering result
# for idx, label in enumerate(cluster_labels):
#     teethList[idx].cluster_label = label

# # Plot original image with red dots at center points
# plt.imshow(image)

# # Plot teeth based on cluster_label attribute
# for label in set(cluster_labels):
#     filtered_teeth = [teeth_obj for teeth_obj in teethList if teeth_obj.cluster_label == label]
#     x_coords = [teeth_obj.calculate_center_coordinate()[0] for teeth_obj in filtered_teeth]
#     y_coords = [teeth_obj.calculate_center_coordinate()[1] for teeth_obj in filtered_teeth]
    
#     # Convert HSV to RGB for matplotlib
#     rgb_color = colorsys.hsv_to_rgb(hue_values[label], 1, 1)
    
#     plt.scatter(x_coords, y_coords, color=rgb_color, marker='o', s=50, label=f'Cluster {label + 1}')

# # Add legend
# plt.legend()

# plt.title('Original Image with Teeth Center Points and K-Means Clustering on X and Y Coordinates (Distinct Hues)')
# plt.show()

# naive approach
# only for face, left, right
# if average area of lower jaw is more than upper then picture is upsidedown
upperJawArea = 0
upperJawTeethCount = 0
lowerJawArea = 0
lowerJawTeethCount = 0

for teeth in teethList:
    if teeth.upperOrLowerJaw == 1:
        upperJawArea += teeth.area
        upperJawTeethCount += 1
    
    else:
        lowerJawArea += teeth.area
        lowerJawTeethCount += 1

avgUpperJawArea = upperJawArea / upperJawTeethCount
avgLowerJawArea = lowerJawArea / lowerJawTeethCount

if avgUpperJawArea > avgLowerJawArea:
    print('picture is in correct orientation')

else:
    print('picture is upside down')