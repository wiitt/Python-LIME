import cv2
import numpy as np
import LIME_functions as LIME

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d

weight_strategy = 3
gamma = 0.8
std_dev = 0.04
img_path = './Photos/HDR/'
output_dir = img_path + 'Output/'
name_list = ['CR', 'BaW', 'SLH', 'LE', 'BoG', 'HC']

for img in name_list:
    image_read = cv2.imread(img_path + img + '.tif', cv2.IMREAD_COLOR)
    print(f'Image: "{img_path + img}.tif", resolution: {image_read.shape}')
    image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) / 255
    illumination_map = np.max(image_rgb, axis=-1)
    updated_ill_map = LIME.update_illumination_map(
        illumination_map, weight_strategy)
    corrected_ill_map = LIME.gamma_correction(np.abs(updated_ill_map), gamma)
    corrected_ill_map = corrected_ill_map[..., np.newaxis]
    new_image = image_rgb / corrected_ill_map
    new_image_w3 = np.clip(new_image, 0, 1).astype("float32")
    denoised_image = LIME.bm3d_yuv_denoising(
        new_image_w3, corrected_ill_map, std_dev)
    ref_image = cv2.imread(img_path + img + '_HDR.jpg', cv2.IMREAD_COLOR)
    ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB) / 255
    downsampled_factor = 100 / min(image_rgb.shape[0:1])
    downsampled_den = cv2.resize(denoised_image, (0, 0), 
                                 fx=downsampled_factor, fy=downsampled_factor)
    downsampled_ref = cv2.resize(ref_image_rgb, (0, 0), 
                                 fx=downsampled_factor, fy=downsampled_factor)
    ref_il_map = np.max(downsampled_ref, axis=-1)
    den_il_map = np.max(downsampled_den, axis=-1)
    print('LOE:', LIME.loss_calculation(ref_il_map, den_il_map), '\n')

    plt.figure(figsize = (12, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_image_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)

    denoised_image_bgr = cv2.cvtColor(denoised_image * 255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_dir + img + '_LIME.jpg', denoised_image_bgr)

plt.show()