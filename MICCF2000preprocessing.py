import cv2
import numpy as np
import os

# Function to apply SRM filters using loaded weights
def apply_srm_filters(image, srm_weights):
    steps = srm_weights.shape[-1]  # Number of filters
    filtered_channels = []

    for step in range(steps):
        filtered_channels_step = []
        for channel in range(3):  # Apply each filter to each color channel
            filter_kernel = srm_weights[:, :, 0, step]
            filtered_image = cv2.filter2D(image[:, :, channel], -1, filter_kernel)
            filtered_channels_step.append(filtered_image)
        
        combined_residual = np.stack(filtered_channels_step, axis=-1)
        filtered_channels.append(combined_residual)

    combined_residual = np.mean(filtered_channels, axis=0)  # Average over all filters
    combined_residual = np.clip(combined_residual, -3, 3)  # Clipping

    # Normalize to the range [0, 255]
    combined_residual = ((combined_residual - combined_residual.min()) / 
                         (combined_residual.max() - combined_residual.min())) * 255.0
    
    return combined_residual.astype(np.uint8)

# Function to preprocess a single image with SRM filters and save it with modified filename
def preprocess_and_save_srm_image(image_path, srm_weights):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Apply SRM filters
    srm_filtered_image = apply_srm_filters(image, srm_weights)
    
    # Derive the new filename with `_srm` suffix
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_srm{ext}"
    
    # Save the SRM filtered image
    cv2.imwrite(output_path, srm_filtered_image)
    print(f'Saved SRM processed image to {output_path}')

def main():
    # Path to the image to be processed
    image_path = 'picture.png'

    # Load SRM weights
    srm_weights = np.load('SRM_Kernels.npy')
    
    # Normalize each filter by its maximum absolute value
    for i in range(srm_weights.shape[-1]):
        max_value = np.max(np.abs(srm_weights[:, :, 0, i]))
        srm_weights[:, :, 0, i] /= max_value

    # Preprocess and save SRM image
    preprocess_and_save_srm_image(image_path, srm_weights)

if __name__ == '__main__':
    main()
