# Feature Engineering & Image Processing

A collection of projects on classic and modern image processing: intensity and color handling, filtering, edge detection, frequency-domain methods, wavelets, and Gabor-based feature extraction. Implementations are in Python using NumPy, OpenCV, SciPy, and scikit-image.

---

## Contents

| Topic | Description |
|-------|-------------|
| [Histogram Matching](#1-histogram-matching) | CDF-based histogram matching for grayscale and color images |
| [Noise & Filtering](#2-noise--filtering) | Denoising (Gaussian, median, bilateral) and edge detection (Sobel, Canny) |
| [Line Detection](#3-line-detection) | Hough transform with geometric width estimation and Otsu binarization |
| [Fourier Domain](#4-fourier-domain) | 2D FFT, magnitude/phase, high-pass filtering, and FFT-based convolution |
| [Wavelets & Keypoints](#5-wavelets--keypoints) | DWT, SIFT keypoints with Shi–Tomasi corner filtering, LL-subband analysis |
| [Gabor & Fingerprints](#6-gabor--fingerprints) | Gabor filter bank on SOCOFing, GUI tool, and classification pipeline |

---

## 1. Histogram Matching

- **Notebook:** `Assigment1_AMohammadshafie.ipynb`
- **What it does:** Matches the intensity distribution of a source image to a reference image using CDF normalization and a proper mapping from source CDF to reference CDF. Supports grayscale and color (with per-channel or grayscale conversion).
- **Highlights:** Correct CDF normalization to [0, 1], reference-driven mapping (not just equalization), and handling of both file paths and NumPy arrays.

---

## 2. Noise & Filtering

- **Folder:** `Assignment 2(noise and filter)/`
- **Notebook:** `Assighnment2_AMohammadshafie.ipynb`
- **What it does:** Adds Gaussian noise, then denoises with Gaussian, median, and bilateral filters. Compares Sobel and Canny edge detection at different thresholds and at multiple resize scales (25%, 50%, 75%).
- **Outputs:** Results are saved under `assignment2_results/` (original, noisy, denoised, and edge images).

**Run:** Place `paris.jpg` in the same folder as the notebook and run all cells. Dependencies: `numpy`, `matplotlib`, `opencv-python`, `scipy`, `pillow`.

---

## 3. Line Detection

- **Folder:** `Assignment3/`
- **Notebook:** `Assignment3_AMohammadshafie.ipynb`
- **What it does:** Detects line segments with the Hough transform, then filters them by **geometric width**: Otsu binarization, perpendicular probing along each segment, median width over samples, and keeping only segments whose width matches a target (within tolerance).
- **Highlights:** Width is measured in the image (perpendicular to the line), not approximated from length.

---

## 4. Fourier Domain

- **Folder:** `Assinment4/`
- **Notebook:** `Assignment4_AMohammadshafie.ipynb`
- **What it does:** Loads a grayscale image, computes 2D FFT, and visualizes magnitude (log-scaled) and phase. Builds a circular high-pass mask in the frequency domain, then implements filtering in two ways: (1) frequency-domain multiplication and (2) **spatial kernel from inverse FFT** of the mask, with **FFT-based convolution** for consistent circular behavior (no zero-padding artifacts).
- **Image:** Uses `paris2.jpg` in the same folder.

---

## 5. Wavelets & Keypoints

- **Folder:** `Assignment5/`
- **Notebook:** `Assignment5_AMohammadshafie.ipynb`
- **What it does:** Applies Discrete Wavelet Transform (DWT), visualizes subbands (LL, LH, HL, HH), and reconstructs the image. Uses SIFT for multi-scale keypoint proposals, then filters by **Shi–Tomasi corner response** (minimum eigenvalue) and keeps the strongest corner-like points. Visualizes these on the original image and on a reconstruction with the LL subband set to zero. OpenCV outputs are converted to RGB for correct display in Matplotlib.
- **Artifacts:** `wavelet_coeffs.npz` and `paris.jpg` are used by the notebook.

---

## 6. Gabor & Fingerprints

- **Folder:** `Project/`
- **Notebooks:** `notebooks/problem.ipynb`, `notebooks/CSCE5222_Gabor_SOCOFing.ipynb`
- **App:** `apps/app_gabor_gui.py` — desktop GUI to load an image, define a Gabor filter bank (orientation and frequency ranges), apply filters, visualize response magnitudes, and export selected response images plus a JSON manifest.

**Pipeline:** Gabor filter bank on fingerprint images (SOCOFing), feature extraction, and classification (e.g., SVM/KNN). See `apps/README_GABOR_GUI.md` for GUI usage.

**Dataset:** SOCOFing is not included in this repo due to size. Download it from the [official source](https://www.kaggle.com/datasets/ruizgara/socofing) and place it under `Project/SOCOFing/` (or adjust paths in the notebooks).

---

## Setup

- **Python:** 3.9+ recommended  
- **Install:**  
  `pip install numpy matplotlib opencv-python scipy pillow scikit-image`

Individual notebooks may require an image file (e.g. `paris.jpg`, `paris2.jpg`, `dal.png`) in the same directory; paths are set at the top of each notebook.

---

## Repository structure

```
.
├── README.md
├── Assigment1_AMohammadshafie.ipynb          # Histogram matching
├── Assignment 2(noise and filter)/           # Noise, filters, edges
├── Assignment3/                              # Hough + width-based line detection
├── Assinment4/                               # Fourier, high-pass, FFT convolution
├── Assignment5/                              # Wavelets, SIFT, Shi–Tomasi corners
└── Project/                                  # Gabor filter bank & SOCOFing
    ├── apps/                                 # Gabor GUI and README
    └── notebooks/                            # Gabor pipeline notebooks
```

---

## Author

**Alireza Mohammadshafie**  
If you use or build on this work, a link back or citation is appreciated.

