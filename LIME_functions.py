import cv2
import numpy as np

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
from typing import Union

def d_sparse_matrices(illumination_map: np.ndarray) -> csr_matrix:
    """Generates Toeplitz matrices of the compatible shape with the given 
    ''illumination_map''
    for computation of a forward difference in both horizontal and vertical 
    directions.

    Returns the shape-(M*N, M*N) arrays of Toeplitz matrices in a compressed 
    sparse row format.

    ## Args:
        illumination_map (numpy.ndarray) : A shape-(M, N) array of maximum 
        intensity values.

    ## Returns:
        d_x_sparse (scipy.sparse.csr_matrix) : A shape-(M*N, M*N) compressed 
        sparse row matrix for calculation of a forward difference 
        in a horizontal direction.

        d_y_sparse (scipy.sparse.csr_matrix) : A shape-(M*N, M*N) compressed 
        sparse row matrix for calculation of a forward difference 
        in a vertical direction.
    """

    image_x_shape = illumination_map.shape[-1]
    image_size = illumination_map.size
    dx_row, dx_col, dx_value = [], [], []
    dy_row, dy_col, dy_value = [], [], []
    # Produces lists of non-zero values and their row and column indeces
    for i in range(image_size - 1):
        if image_x_shape + i < image_size:
            dy_row += [i, i]
            dy_col += [i, image_x_shape + i]
            dy_value += [-1, 1]
        if (i+1) % image_x_shape != 0 or i == 0:
            dx_row += [i, i]
            dx_col += [i, i+1]
            dx_value += [-1, 1]
    # Creates compressed sparse row matrices of a required shape
    # based on provided values and their indeces
    d_x_sparse = csr_matrix((dx_value, (dx_row, dx_col)), 
                            shape = (image_size, image_size))
    d_y_sparse = csr_matrix((dy_value, (dy_row, dy_col)), 
                            shape = (image_size, image_size))

    return d_x_sparse, d_y_sparse


def partial_derivative_vectorized(
        input_matrix: np.ndarray,
        toeplitz_sparse_matrix: csr_matrix
        ) -> np.ndarray:
    """Calculates a partial derivative of an ''input_matrix'' with a given 
    ''toeplitz_sparse_matrix''.

    Returns the shape-(M, N) array of derivative values.

    ## Args:
        input_matrix (numpy.ndarray) : A shape-(M, N) array.

        toeplitz_sparse_matrix (scipy.sparse.csr_matrix) : A shape-(M*N, M*N) 
        compressed sparse row matrix for calculation of a difference 
        in a specified direction.

    ## Returns:
        p_derivative (numpy.ndarray) : A shape-(M, N) array of derivative 
        values.
    """

    input_size = input_matrix.size
    output_shape = input_matrix.shape
    # Vectorizes the input matrix producing a shape-(M*N, 1) vector
    vectorized_matrix = input_matrix.reshape((input_size, 1))
    # Calculates values of partial derivatives with multiplication of the 
    # vectorized matrix by the specific Toeplitz matrix in a compressed
    # sparse row format
    matrices_product = toeplitz_sparse_matrix * vectorized_matrix
    # Reverts vectorized matrix of partial derivatives to a shape
    # of the input matrix
    p_derivative = matrices_product.reshape(output_shape)

    return p_derivative


def gaussian_weight(
        grad: np.ndarray,
        size: int,
        sigma: Union[int, float],
        epsilon: float
        ) -> np.ndarray:
    """Initializes weight matrix according to the third wieght strategy of the 
    original LIME paper.

    Returns the shape-(M, N) array of weight values.

    ## Args:
        grad (numpy.ndarray) : A shape-(M, N) array of partial gradient values.

        size (int) : An odd value which charactarizes the size of a Gaussian 
        kernel.

        sigma (int or float) : A standard deviation value of a Gaussian kernel.

        epsilon (float) : A small value which prevents division by zero 
        occurrences.

    ## Returns:
        weights (numpy.ndarray) : A shape-(M, N) array of weights.
    """

    radius=int((size-1)/2)
    denominator = epsilon + gaussian_filter(np.abs(grad), sigma, radius=radius, mode='constant')
    weights = gaussian_filter(1 / denominator, sigma, radius=radius, mode='constant')

    return weights


def initialize_weights(
        ill_map: np.ndarray,
        strategy_n: int,
        epsilon: float = 0.001
        ) -> np.ndarray:
    """Initializes weight matrices according to a chosen strategy of 
    the original LIME paper. Then updates and vectorizes these weight matrices 
    preparing them to be used for calculation of a new illumination map. 

    Returns the shape-(M, N) arrays of weight values with regard to horizontal 
    and vertical directions.

    ## Args:
        ill_map (numpy.ndarray) : A shape-(M, N) array of maximum intensity 
        values.

        strategy_n (int) : A number of a selected strategy for weigth 
        initialization. Could be 1, 2 or 3.

        epsilon (float) : A small value which prevents division by zero 
        occurrences.

    ## Returns:
        flat_w_x (numpy.ndarray) : A shape-(1, M*N) vectorized array of 
        updated weights with regard to horizontal direction.

        flat_w_y (numpy.ndarray) : A shape-(1, M*N) vectorized array of 
        updated weights with regard to vertical direction.
    """

    # Initializes weight matrices according to a chosen strategy
    if strategy_n == 1:
        print('Weight generation strategy: 1')
        weights = np.ones(ill_map.shape)
        weights_x = weights
        weights_y = weights
    elif strategy_n == 2:
        print('Weight generation strategy: 2')
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = 1 / (np.abs(grad_t_x) + epsilon)
        weights_y = 1 / (np.abs(grad_t_y) + epsilon)
    else:
        sigma = 2
        size = 15
        print('Weight generation strategy: 3')
        print(f'Strategy parameters: sigma = {sigma}, kernel size = {size}')
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = gaussian_weight(grad_t_x, size, sigma, epsilon)
        weights_y = gaussian_weight(grad_t_y, size, sigma, epsilon)
    # Modifies and transforms weight matrices in a vector form
    modified_w_x = weights_x / (np.abs(grad_t_x) + epsilon)
    modified_w_y = weights_y / (np.abs(grad_t_y) + epsilon)
    flat_w_x = modified_w_x.flatten()
    flat_w_y = modified_w_y.flatten()

    return flat_w_x, flat_w_y


def update_illumination_map(
        ill_map: np.ndarray,
        weight_strategy: int = 3
        ) -> np.ndarray:
    """Updates the initial illumination map according to a sped-up solver of 
    the original LIME paper.

    Returns the shape-(M, N) updated illumination map array.

    ## Args:
        ill_map (numpy.ndarray) : A shape-(M, N) array of maximum intensity 
        values.

        weight_strategy (int) : A number of a selected strategy for weigth 
        initialization. Could be 1, 2 or 3.

    ## Returns:
        (numpy.ndarray) : A shape-(M, N) array of updated values of 
        illumination map.
    """

    # Vectorizes initial illumination map
    vectorized_t = ill_map.reshape((ill_map.size, 1))
    epsilon = 0.001
    alpha = 0.15
    # Generates Toeplitz matrices of for computation of a forward difference 
    # in both horizontal and vertical directions
    d_x_sparse, d_y_sparse = d_sparse_matrices(ill_map)
    # Initializes vectorized weight matrices according to a chosen strategy
    flatten_wiegths_x, flatten_wiegths_y = initialize_weights(
       ill_map, weight_strategy, epsilon)
    # Constructs diagonal matrices from vectorized weights
    diag_weights_x = diags(flatten_wiegths_x)
    diag_weights_y = diags(flatten_wiegths_y)
    # Updates the illumination map by solving the equation (19) of 
    # the original LIME paper
    x_term = d_x_sparse.transpose() * diag_weights_x * d_x_sparse
    y_term = d_y_sparse.transpose() * diag_weights_y * d_y_sparse
    identity = diags(np.ones(x_term.shape[0]))
    matrix = identity + alpha * (x_term + y_term)
    updated_t = spsolve(csr_matrix(matrix), vectorized_t)
    print('Solved:', type(updated_t), '\n')

    return updated_t.reshape(ill_map.shape)


def gamma_correction(
        ill_map: np.ndarray,
        gamma: Union[int, float]
        ) -> np.ndarray:
    """Performes gamma correction of the initial illumination map with 
    a given ''gamma'' coefficient.

    Returns the shape-(M, N) corrected illumination map array.

    ## Args:
        ill_map (numpy.ndarray) : A shape-(M, N) array of maximum intensity 
        values.

        gamma (int or float) : A value of gamma correction coefficient.

    ## Returns:
        (numpy.ndarray) : A shape-(M, N) array of corrected values of 
        the illumination map.
    """

    return ill_map ** gamma


def bm3d_yuv_denoising(
        image: np.ndarray,
        cor_ill_map: np.ndarray,
        std_dev: Union[int, float]=0.02
        ) -> np.ndarray:
    """Performes denoising of an image Y color channel with B3MD algorithm and 
    corrects its brigghtness with an updated illumination map.

    Returns a shape-(M, N) denoised image with corrected brightness in which
    pixel intensities exceeding 1 are clipped.

    ## Args:
        image (numpy.ndarray) : A shape-(3, M, N) initial image.

        cor_ill_map (numpy.ndarray) : A shape-(M, N) array of 
        corrected intensity values.

        std_dev (int or float) : A value of standard deviation parameter for 
        the BM3D algorithm.

    ## Returns:
        (numpy.ndarray) : A shape-(M, N) denoised image with a corrected 
        illumination map.
    """

    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0]
    denoised_y_ch = bm3d(y_channel, std_dev)
    image_yuv[:, :, 0] = denoised_y_ch
    denoised_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    recombined_image = image * cor_ill_map + denoised_rgb * (1 - cor_ill_map)

    return np.clip(recombined_image, 0, 1).astype("float32")


def is_image(file_name: str) -> bool:
    """Checks if a file is of 'bmp', 'jpg', 'png' or 'tif' format.

    Returns True if a file name ends with any of these formats, and False is 
    returned otherwise.

    ## Args:
        file_name (str) : A string representing a file name.

    ## Returns:
        bool_value (bool) : A boolean value answering if the provided file 
        name is of the given four formats.   
    """

    bool_value = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']

    return bool_value


def loss_calculation(
        reference_image: np.ndarray,
        refined_image: np.ndarray
        ) -> float:
    """Calculates the lightness order error (LOE) metric comparing pixel 
    intensities of a refined image with their reference counterparts.

    Returns a calculated value of the LOE metric.

    ## Args:
        reference_image (numpy.ndarray) : A shape-(3, M, N) reference image 
        which is considered as ground truth.

        refined_image (numpy.ndarray) : A shape-(3, M, N) refined image.

    ## Returns:
        (float) : A calculated value of the LOE metric.
    """

    v_shape, h_shape = reference_image.shape
    n_pixels = reference_image.size
    loss = 0

    for v_pixel in range(v_shape-1):
        for h_pixel in range(h_shape-1):
            bool_term_ini = reference_image <= \
                  reference_image[v_pixel, h_pixel]
            bool_term_ref = refined_image <= refined_image[v_pixel, h_pixel]
            xor_term = np.logical_xor(bool_term_ini, bool_term_ref)
            loss += np.sum(xor_term)

    return loss / (n_pixels * 1000)