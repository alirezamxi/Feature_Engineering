## Gabor Filter Bank GUI

This desktop GUI lets you:
- Load a fingerprint (or any) image
- Define a Gabor filter bank by orientation and frequency ranges (with steps)
- Apply the filters and visualize response magnitudes
- Select a subset of filters and save the corresponding response images
- Export a JSON manifest with selected filter parameters and preprocessing info

### Requirements
- Python 3.9+ recommended
- numpy, matplotlib, scikit-image
  - These are already used in your notebooks. If missing: `pip install numpy matplotlib scikit-image`

### Run

```bash
python apps/app_gabor_gui.py
```

### Usage
1. File → Load Image… (supports PNG/JPG/BMP/TIF/GIF). Adjust resize H×W if needed, then Apply Resize.
2. Set Gabor ranges:
   - Orientations (degrees): start, stop, step (e.g., 0 to 180 step 45)
   - Frequencies: start, stop, step (e.g., 0.1 to 0.3 step 0.1)
   - Click “Create Filter Bank” to generate the list.
3. Click “Apply Filters to Image” to compute response magnitudes for all filters.
4. Multi-select any subset in the list and click “Preview Selected” to view.
5. Click “Save Selected…” to write:
   - Response images as PNG files (uint8 visualization)
   - A `*_selected_filters.json` manifest with:
     - source image path
     - resize dimensions
     - selected filter parameters (`theta_deg`, `frequency`)

Notes:
- Visualization is min–max normalized for display; numerical arrays aren’t saved—only images and JSON.
- If you need to export raw arrays for research, we can add NumPy `.npy` export.



