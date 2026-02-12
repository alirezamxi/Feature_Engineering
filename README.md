# Image Processing & Feature Engineering (Python)

Practical projects exploring classic image processing and feature extraction techniques, implemented in Python with Jupyter notebooks and a small desktop GUI tool. Topics include histogram-based intensity mapping, denoising and edge detection, Hough line detection with geometric width filtering, Fourier-domain filtering, wavelets, and Gabor texture features with a lightweight classification pipeline.

---

## Highlights

- **End-to-end notebooks**: each notebook runs top-to-bottom and produces visual outputs.
- **Hands-on feature extraction**: histogram/CDF mapping, Gabor response statistics, wavelet subbands, corner-like keypoint filtering.
- **Reproducible artifacts**: saved images (e.g. edge maps) and exported filter parameters (JSON) for downstream use.

---

## Projects

### 1) Histogram matching & basic image operations

- **Notebook**: `Assigment1_AMohammadshafie.ipynb`
- **What’s inside**
  - Color-space exploration (HSV split) and basic resizing/cropping steps.
  - Grayscale histogram visualization.
  - **Histogram matching** using CDF normalization and a mapping from source CDF to reference CDF.
  - Utility that accepts **file paths or NumPy arrays** as inputs.

### 2) Noise, denoising, and edges

- **Folder**: `Assignment 2(noise and filter)/`
- **Notebook**: `Assighnment2_AMohammadshafie.ipynb`
- **What’s inside**
  - Add Gaussian noise.
  - Denoise with Gaussian / median / bilateral filters.
  - Edge detection comparisons (Sobel vs Canny), threshold sweeps, and resizing experiments.
  - Outputs saved under `Assignment 2(noise and filter)/assignment2_results/`.

### 3) Hough line detection with width filtering

- **Folder**: `Assignment3/`
- **Notebook**: `Assignment3_AMohammadshafie.ipynb`
- **What’s inside**
  - Line segment detection using `cv2.HoughLinesP` with a minimum length.
  - **Geometric line width estimation**:
    - Binarization via **Otsu thresholding** (`THRESH_BINARY_INV + THRESH_OTSU`).
    - Per-segment width measured by probing **perpendicular** to the line direction at multiple sample points.
    - Robust width estimate via the **median** of measured widths.
  - Final filtering by both **length** and **width tolerance**.

### 4) Fourier analysis & high‑pass filtering

- **Folder**: `Assinment4/`
- **Notebook**: `Assignment4_AMohammadshafie.ipynb`
- **What’s inside**
  - 2D FFT, centered spectrum visualization, and **magnitude/phase** display (log magnitude for visibility).
  - Circular **high‑pass mask** in the frequency domain (cutoff ratio parameter).
  - Reconstruction via inverse FFT.
  - Spatial-domain equivalent filter obtained by **IFFT of the frequency mask**, applied using **FFT-based convolution** (circular convolution behavior).

### 5) Wavelets & keypoints

- **Folder**: `Assignment5/`
- **Notebook**: `Assignment5_AMohammadshafie.ipynb`
- **What’s inside**
  - 2D Haar DWT (`pywt.dwt2`) and visualization of **LL / LH / HL / HH** subbands.
  - Reconstruction from subbands and saved coefficients (`wavelet_coeffs.npz`).
  - SIFT keypoint detection (`cv2.SIFT_create`) and visualization.
  - Filtering SIFT keypoints using a **Shi–Tomasi** corner response (`cv2.cornerMinEigenVal`) and ranking strongest responses.
  - Visualization on both the original image and a reconstruction with **LL removed** (LL = 0).

### 6) Gabor texture features for fingerprint alteration detection (+ GUI)

- **Folder**: `Project/`
- **Notebook**: `Project/notebooks/CSCE5222_Gabor_SOCOFing.ipynb`
- **What’s inside**
  - Gabor filter bank over multiple orientations and frequencies.
  - Feature extraction from response magnitudes using simple summary statistics (e.g., mean/std).
  - Baseline classifiers (SVM and k‑NN) with scaling and train/test split.

- **GUI tool**: `Project/apps/app_gabor_gui.py`
  - Load an image (robust handling for grayscale/RGB/RGBA).
  - Define orientation/frequency ranges to build a filter bank.
  - Apply filters, preview response magnitudes, and **export selected responses + JSON parameters**.
  - See `Project/apps/README_GABOR_GUI.md` for usage details.

---

## Setup

- **Python**: 3.9+ recommended
- **Install dependencies**

```bash
pip install numpy matplotlib pillow opencv-python scipy scikit-image pywavelets scikit-learn pandas seaborn
```

- **Run notebooks**

```bash
jupyter lab
```

Most notebooks expect an example image in the same folder (e.g. `paris.jpg`, `paris2.jpg`, `dal.png`).

---

## Dataset note (SOCOFing)

The SOCOFing dataset is not committed to this repo (large). Download it from Kaggle and place it at `Project/SOCOFing/` (or update paths in the notebook):

- `https://www.kaggle.com/datasets/ruizgara/socofing`

---

## Repository structure

```
.
├── README.md
├── Assigment1_AMohammadshafie.ipynb
├── Assignment 2(noise and filter)/
├── Assignment3/
├── Assinment4/
├── Assignment5/
└── Project/
    ├── apps/
    └── notebooks/
```

---

## Author

**Alireza Mohammadshafie**

