import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.signal import correlate2d

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def apply_blur(image, kernel_size, sigma):
    """Apply Gaussian blur to an image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    return blurred_image

def wiener_filter(blurred_image, kernel, noise_var):
    """Apply Wiener filter to deblur the image."""
    kernel_fft = np.fft.fft2(kernel, s=blurred_image.shape)
    blurred_fft = np.fft.fft2(blurred_image)
    kernel_conj = np.conj(kernel_fft)
    
    wiener_filter_fft = kernel_conj / (kernel_fft * kernel_conj + noise_var)
    deblurred_fft = wiener_filter_fft * blurred_fft
    deblurred_image = np.fft.ifft2(deblurred_fft)
    return np.abs(deblurred_image)

# Load and prepare the image
# image = cv2.imread('outputs/101_2_face.jpg', 0)  # Load in grayscale
# image = image / 255.0  # Normalize to 0-1

# # Apply blur
# kernel_size = 5
# sigma = 1.0
# blurred_image = apply_blur(image, kernel_size, sigma)

# Estimate noise variance manually (or through more elaborate means)
# noise_var = 0.01

# Deblur the image
# deblurred_image = wiener_filter(blurred_image, gaussian_kernel(kernel_size, sigma), noise_var)

# Display results using OpenCV (or any other tool you prefer)
# cv2.imshow('Original Image', image)
# cv2.imshow('Blurred Image', blurred_image)
# cv2.imshow('Deblurred Image', deblurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
