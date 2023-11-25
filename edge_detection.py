import cv2
import numpy as np
import os
import sys
import math

def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel = np.uint8(sobel / sobel.max() * 255)
    return sobel

def apply_canny(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges

def apply_zero_crossing(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    result = np.zeros_like(laplacian)
    rows, cols = laplacian.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = laplacian[i - 1:i + 2, j - 1:j + 2]
            if (np.sign(patch) == -1).any() and (np.sign(patch) == 1).any():
                result[i, j] = 255
    result = result.astype(np.uint8)
    return result

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse_value))

def edge_detection(input_folder, output_folder, reference_folder=None):
    if not os.path.exists(input_folder):
        print(f"The input folder {input_folder} does not exist.")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(os.path.join(output_folder, 'results.txt'), 'w') as results_file:
        for image_file in image_files:
            print(f"Processing {image_file}...")
            input_path = os.path.join(input_folder, image_file)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error reading {input_path}. Skipping.")
                continue

            reference_image = None
            if reference_folder:
                reference_path = os.path.join(reference_folder, image_file)
                reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
                if reference_image is None:
                    print(f"Reference image {reference_path} not found. Skipping MSE and PSNR calculations.")
                    results_file.write(f"{image_file}, reference not found, MSE:, PSNR:\n")
                    continue

            for method in ['sobel', 'canny', 'zero_crossing']:
                processed_image = globals()[f'apply_{method}'](image)
                output_path = os.path.join(output_folder, f"{method}_{image_file}")
                cv2.imwrite(output_path, processed_image)
                print(f"Saved {output_path}")

                if reference_folder:
                    current_mse = mse(reference_image, processed_image)
                    current_psnr = psnr(reference_image, processed_image)
                    print(f"{method} - {image_file} - MSE: {current_mse:.2f}, PSNR: {current_psnr:.2f}")
                    results_file.write(f"{image_file}, {method}, MSE: {current_mse:.2f}, PSNR: {current_psnr:.2f}\n")
    
    # print("avg psnr:", avg_psnr)
    # print("avg_mse:",avg_mse)
    print("Edge detection completed for all images.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python edge_detection.py <input_folder> <output_folder> <reference_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    reference_folder = sys.argv[3]  # Take the reference folder from command line
    edge_detection(input_folder, output_folder, reference_folder)
