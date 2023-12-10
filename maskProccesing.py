import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TeethObject:
    def __init__(self, mask):
        self.mask = mask
        self.center, self.area = self.calculate_center_and_area(mask)

    @staticmethod
    def calculate_center_and_area(mask):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Calculate the moments of the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)

            # Calculate the centroid (center) of the largest contour
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                center = (cX, cY)
            else:
                center = None

            # Calculate the area of the largest contour
            area = cv2.contourArea(largest_contour)
        else:
            center = None
            area = 0

        return center, area


def change_color(img, target_color):
    # Convert the image to NumPy array
    img_array = np.array(img)

    # Create a mask for non-colored pixels
    mask = np.all(img_array[:, :, :3] != target_color, axis=-1)

    # Display the mask as a binary image
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

    # Set non-colored pixels to black
    img_array[mask, :3] = [0, 0, 0]

    # Create a new image from the NumPy array
    new_img = Image.fromarray(img_array)

    # Display the new image
    plt.subplot(1, 3, 2)
    plt.imshow(new_img)
    plt.title('Modified Image')

    # Display the target color
    plt.subplot(1, 3, 3)
    plt.imshow([[target_color]])
    plt.title('Target Color')

    plt.show()

    color = get_unique_colors(np.array(new_img))
    print(color)

    return new_img


def get_unique_colors(image):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)

    return unique_colors

def get_unique_colors_old(image_path):
    image = cv2.imread(image_path)
    pixels = image.reshape((-1, 3))
    unique_colors_set = set(map(tuple, pixels))
    unique_colors_list = list(unique_colors_set)
    return unique_colors_list

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


# Example usage
image_path = 'result/1/mask/mask_IMG_2898.png'
image = Image.open(image_path)

colors = get_unique_colors(image)
# won't include black as black is backround
print(len(colors)-1)
# print(colors)

for color in colors:
    black_color = np.array([0, 0, 0])

    # Check if the color is not black
    if not np.array_equal(color, black_color):
        change_color(image, color)
