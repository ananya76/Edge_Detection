### README

# Edge Detection Comparative Study

## Abstract

Edge detection is a fundamental operation in image processing and pattern recognition, serving as a cornerstone in various applications. Edges delineate the boundaries of objects within images, reducing data size without compromising essential information. They also play a crucial role in image enhancement, recognition, restoration, and compression. Understanding edge positions is pivotal in these processes. This study provides an understanding of different methods of edge detection and a detailed comparative study on four edge detection algorithms, namely Sobel, Canny, Zero Crossing, and Holistically Nested Edge Detection (HED). The study aims to evaluate their performance in terms of image processing, encompassing broad and influential domains. We evaluate the performance of these algorithms by analyzing results through metrics like mean square error and peak signal-to-noise ratio. The findings in this study contribute to the broader understanding of the most suitable edge detection methods for various applications, emphasizing the critical role edges play in human image perception and data processing.

## Dataset

The BSDS300 dataset was used for the comparative study. This dataset contains a variety of images that are commonly used for benchmarking edge detection algorithms.

## Edge Detection Algorithms

The following edge detection algorithms were compared in this study:

1. **Sobel**: An edge detection algorithm that computes the gradient magnitude of the image intensity at each pixel, highlighting regions of high spatial frequency.
2. **Canny**: A multi-stage edge detection algorithm that uses a combination of Gaussian filtering, gradient computation, non-maximum suppression, and hysteresis thresholding.
3. **Zero Crossing**: An edge detection method that detects edges by finding zero crossings in the second derivative of the image intensity.
4. **Holistically Nested Edge Detection (HED)**: A deep learning-based approach that uses convolutional neural networks to detect edges at multiple scales and levels of abstraction.

## Performance Metrics

The performance of the edge detection algorithms was evaluated using the following metrics:

- **Mean Square Error (MSE)**: Measures the average squared difference between the original and the detected edge image.
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.

## Results

The results of the comparative study are presented in the study report, highlighting the strengths and weaknesses of each algorithm in various image processing scenarios.

## Repository Structure

- `input/`: Contains the BSDS300 dataset images.
- `edge_detection/`: Contains the implementation scripts for the edge detection algorithms
- `README.md`: This file.

## Contributions

Contributions to improve the algorithms or add new edge detection methods are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on the GitHub repository or contact the authors.
