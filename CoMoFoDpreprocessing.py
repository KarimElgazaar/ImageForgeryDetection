import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import random 
import pickle
import os

# Function to convert an image to an ELA image
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

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

# Prepare image for ELA
def prepare_ela_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path, 90).resize(image_size)
    return np.array(ela_image).flatten() / 255.0

# Prepare image for SRM using loaded weights (apply SRM before resizing)
def prepare_srm_image(image_path, srm_weights, image_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Apply SRM filters before resizing
    combined_residual = apply_srm_filters(image, srm_weights)
    
    # Resize the SRM filtered image
    resized_residual = cv2.resize(combined_residual, image_size)

    return resized_residual.flatten() / 255.0

# Preprocess images with both ELA and SRM
def preprocess_images(base_path, label, srm_weights, image_size=(128, 128), max_images=None):
    X_ela = []
    X_srm = []
    Y = []
    image_count = 0
    file_log = []
    
    for dirname, _, filenames in os.walk(base_path):
        filenames.sort()
        for filename in filenames:
            if filename.endswith(('jpg', 'png', 'jpeg')):
                full_path = os.path.join(dirname, filename)
                X_ela.append(prepare_ela_image(full_path, image_size=image_size))
                X_srm.append(prepare_srm_image(full_path, srm_weights, image_size=image_size))
                Y.append(label)
                file_log.append(full_path)
                image_count += 1
                if len(Y) % 500 == 0:
                    print(f'Processing {len(Y)} images')
                if max_images and image_count >= max_images:
                    return X_ela, X_srm, Y, file_log
    return X_ela, X_srm, Y, file_log

# Save preprocessed data
def save_preprocessed_data(X_ela, X_srm, Y, file_log, filename='CoMoFoDdata.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((X_ela, X_srm, Y, file_log), f)

def main():
    image_size = (128, 128)

    real_path =
    fake_path = 

    # Load SRM weights
    srm_weights = np.load('SRM_Kernels.npy')
    
    # Normalize each filter by its maximum absolute value
    for i in range(srm_weights.shape[-1]):
        max_value = np.max(np.abs(srm_weights[:, :, 0, i]))
        srm_weights[:, :, 0, i] /= max_value

    X_real_ela, X_real_srm, Y_real, log_real = preprocess_images(real_path, 1, srm_weights, image_size=image_size)
    X_fake_ela, X_fake_srm, Y_fake, log_fake = preprocess_images(fake_path, 0, srm_weights, image_size=image_size)
    
    X_ela = X_real_ela + X_fake_ela
    X_srm = X_real_srm + X_fake_srm
    Y = Y_real + Y_fake
    log = log_real + log_fake
    
    combined = list(zip(X_ela, X_srm, Y, log))
    random.shuffle(combined)    
    X_ela[:], X_srm[:], Y[:], log[:] = zip(*combined)

    print(f'Total images processed: {len(X_ela)}')

    save_preprocessed_data(X_ela, X_srm, Y, log, filename='CoMoFoDdata.pkl')

if __name__ == '__main__':
    main()
