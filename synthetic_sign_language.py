import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# Parameters
num_classes = 5
num_samples_per_class = 1000
img_size = 64

# Create directories for synthetic dataset
dataset_dir = 'synthetic_sign_language'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    for i in range(num_classes):
        os.makedirs(os.path.join(dataset_dir, str(i)))

# Generate synthetic images
def generate_image(label, img_size):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    if label == 0:  # Horizontal line
        cv2.line(img, (0, img_size//2), (img_size, img_size//2), 255, 3)
    elif label == 1:  # Vertical line
        cv2.line(img, (img_size//2, 0), (img_size//2, img_size), 255, 3)
    elif label == 2:  # Diagonal line
        cv2.line(img, (0, 0), (img_size, img_size), 255, 3)
    elif label == 3:  # Circle
        cv2.circle(img, (img_size//2, img_size//2), img_size//4, 255, 3)
    elif label == 4:  # Rectangle
        cv2.rectangle(img, (img_size//4, img_size//4), (3*img_size//4, 3*img_size//4), 255, 3)
    return img

# Save synthetic images
for label in range(num_classes):
    for i in range(num_samples_per_class):
        img = generate_image(label, img_size)
        img_path = os.path.join(dataset_dir, str(label), f'{i}.png')
        cv2.imwrite(img_path, img)

# Display some examples
fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
for label in range(num_classes):
    img = generate_image(label, img_size)
    axes[label].imshow(img, cmap='gray')
    axes[label].set_title(f'Class {label}')
plt.show()
