import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TeethObject:
    def __init__(self, mask):
        self.mask = mask
        self.area = self.calculate_area()

        print(self.area)

    def calculate_area(self):
        return cv2.countNonZero(self.mask)


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


image_path = 'result/1/mask/mask_IMG_2898.png'
image = Image.open(image_path)

teethList = []

colors = get_unique_colors(image)
# won't include black as black is backround
print(len(colors)-1)
# print(colors)

for color in colors:
    black_color = np.array([0, 0, 0])

    # Check if the color is not black
    if not np.array_equal(color, black_color):
        # theres some bug with mask so use new_img for now
        _, new_img = get_color_mask(image, color)

        # convert new_img to binary image
        binary_image = binarizeImage(new_img)
        # plt.imshow(binary_image, cmap='gray')
        # plt.show()

        teethList.append(TeethObject(binary_image))
