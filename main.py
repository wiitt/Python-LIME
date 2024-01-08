import cv2
import numpy as np
import LIME_functions as LIME

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d

folder = './Photos/ExDark/'
output_dir = folder + 'Output/'
weight_strategy = 3
gamma = 0.4
std_dev = 0.04

onlyimages = [f for f in listdir(folder) if LIME.is_image(f)]
for image in onlyimages:
    img_path = folder + image
    image_read = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(f'Image: {img_path}, resolution: {image_read.shape}')
    image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) / 255
    illumination_map = np.max(image_rgb, axis=-1)
    updated_ill_map = LIME.update_illumination_map(
        illumination_map, weight_strategy)
    corrected_ill_map = LIME.gamma_correction(np.abs(updated_ill_map), gamma)
    corrected_ill_map = corrected_ill_map[..., np.newaxis]
    new_image = image_rgb / corrected_ill_map
    new_image = np.clip(new_image, 0, 1).astype("float32")
    denoised_image = LIME.bm3d_yuv_denoising(
        new_image, corrected_ill_map, std_dev)
    plt.figure(figsize = (12, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)
    denoised_image_bgr = cv2.cvtColor(denoised_image * 255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_dir + 'LIME_' + image, denoised_image_bgr)

plt.show()