import cv2
import numpy as np

image = cv2.imread('outputs/101_2_face.jpg')

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(image, 0 , sharpen_kernel)

deblurred = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
# Example usage
image_path = 'outputs/101_2_face.jpg'
# deblurred_image = deblur_image(image_path)

# Display results using OpenCV
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow("deblureed", deblurred)
cv2.waitKey ()


# cv2.imshow('Deblurred Image', np.uint8(deblurred_image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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
