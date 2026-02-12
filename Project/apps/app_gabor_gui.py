#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gabor Filter Bank GUI

Features:
- Load a single image (any common format), robust handling of grayscale/RGB/RGBA
- Specify Gabor parameter ranges with steps to build a filter bank
- Apply the bank to the loaded image and visualize responses
- Select a subset of filters (keeps parameters) as the featured set
- Save selected response images and a JSON with the chosen parameters

Dependencies: numpy, matplotlib, scikit-image
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage import io, color, transform, img_as_ubyte
from skimage.filters import gabor

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ---------------------------
# Data structures
# ---------------------------
@dataclass(frozen=True)
class GaborParams:
	theta_deg: float
	frequency: float

	def key(self) -> str:
		return f"θ={self.theta_deg:g}°, f={self.frequency:g}"


# ---------------------------
# Utility functions
# ---------------------------
SUPPORTED_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


def load_gray_resized(path: Path, size: Tuple[int, int]) -> np.ndarray:
	"""Load an image robustly and return grayscale float in [0, 1] resized to size (h, w)."""
	img = io.imread(path.as_posix())
	if img.ndim == 3:
		# Handle RGBA, RGB, or single-channel stacks
		if img.shape[2] == 4:
			img = color.rgba2rgb(img)
			img_gray = color.rgb2gray(img)
		elif img.shape[2] == 3:
			img_gray = color.rgb2gray(img)
		elif img.shape[2] == 1:
			img_gray = img[..., 0]
		else:
			img_gray = color.rgb2gray(img[..., :3])
	else:
		# already single channel
		img_gray = img.astype(np.float32)
	# resize with anti-aliasing, keep range
	img_resized = transform.resize(
		img_gray, size, anti_aliasing=True, mode="reflect", preserve_range=True
	)
	# normalize to [0, 1]
	img_resized = np.asarray(img_resized, dtype=np.float32)
	min_v, max_v = float(np.min(img_resized)), float(np.max(img_resized))
	if max_v > min_v:
		img_resized = (img_resized - min_v) / (max_v - min_v)
	else:
		img_resized = np.zeros_like(img_resized, dtype=np.float32)
	return img_resized


def gabor_magnitude(image_gray: np.ndarray, theta_deg: float, frequency: float) -> np.ndarray:
	real, imag = gabor(image_gray, theta=np.deg2rad(theta_deg), frequency=frequency)
	mag = np.hypot(real, imag)
	return mag


def to_display_image(arr: np.ndarray) -> np.ndarray:
	"""Normalize a float array to 0-255 uint8 for saving/visualization."""
	arr = np.asarray(arr, dtype=np.float32)
	a_min = float(np.min(arr))
	a_max = float(np.max(arr))
	if a_max > a_min:
		normalized = (arr - a_min) / (a_max - a_min)
	else:
		normalized = np.zeros_like(arr, dtype=np.float32)
	return img_as_ubyte(normalized)


# ---------------------------
# Main Application
# ---------------------------
class GaborGUI(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("Gabor Filter Bank GUI")
		self.geometry("1200x800")

		# State
		self.loaded_image_path: Path | None = None
		self.original_image: np.ndarray | None = None  # grayscale [0,1], resized
		self.target_size: Tuple[int, int] = (256, 256)

		self.filter_bank: List[GaborParams] = []
		self.responses: Dict[GaborParams, np.ndarray] = {}

		# UI
		self._build_menu()
		self._build_layout()
		self._bind_events()

	def _build_menu(self) -> None:
		menubar = tk.Menu(self)
		filemenu = tk.Menu(menubar, tearoff=0)
		filemenu.add_command(label="Load Image...", command=self.on_load_image)
		filemenu.add_separator()
		filemenu.add_command(label="Quit", command=self.destroy)
		menubar.add_cascade(label="File", menu=filemenu)
		self.config(menu=menubar)

	def _build_layout(self) -> None:
		# Main Paned layout: left controls, right viewer
		self.columnconfigure(0, weight=0)
		self.columnconfigure(1, weight=1)
		self.rowconfigure(0, weight=1)

		self.left = ttk.Frame(self, padding=10)
		self.left.grid(row=0, column=0, sticky="ns")
		self.right = ttk.Frame(self, padding=10)
		self.right.grid(row=0, column=1, sticky="nsew")
		self.right.rowconfigure(1, weight=1)
		self.right.columnconfigure(0, weight=1)

		# ---- Left controls
		section_img = ttk.LabelFrame(self.left, text="Image", padding=8)
		section_img.grid(row=0, column=0, sticky="ew", pady=(0, 10))
		self.btn_load = ttk.Button(section_img, text="Load Image...", command=self.on_load_image)
		self.btn_load.grid(row=0, column=0, sticky="ew")
		ttk.Label(section_img, text="Resize (H x W):").grid(row=1, column=0, sticky="w", pady=(8, 0))
		size_frame = ttk.Frame(section_img)
		size_frame.grid(row=2, column=0, sticky="w")
		self.var_h = tk.StringVar(value=str(self.target_size[0]))
		self.var_w = tk.StringVar(value=str(self.target_size[1]))
		ttk.Entry(size_frame, width=6, textvariable=self.var_h).grid(row=0, column=0)
		ttk.Label(size_frame, text="x").grid(row=0, column=1, padx=4)
		ttk.Entry(size_frame, width=6, textvariable=self.var_w).grid(row=0, column=2)
		ttk.Button(section_img, text="Apply Resize", command=self.on_apply_resize).grid(row=3, column=0, pady=(6, 0), sticky="ew")

		section_bank = ttk.LabelFrame(self.left, text="Gabor Filter Bank", padding=8)
		section_bank.grid(row=1, column=0, sticky="ew", pady=(0, 10))
		# Orientations
		ttk.Label(section_bank, text="Orientation (deg): start, stop, step").grid(row=0, column=0, sticky="w")
		self.var_theta_start = tk.StringVar(value="0")
		self.var_theta_stop = tk.StringVar(value="180")
		self.var_theta_step = tk.StringVar(value="45")
		row1 = ttk.Frame(section_bank); row1.grid(row=1, column=0, sticky="w", pady=(2, 6))
		ttk.Entry(row1, width=6, textvariable=self.var_theta_start).grid(row=0, column=0)
		ttk.Label(row1, text="to").grid(row=0, column=1, padx=4)
		ttk.Entry(row1, width=6, textvariable=self.var_theta_stop).grid(row=0, column=2)
		ttk.Label(row1, text="step").grid(row=0, column=3, padx=4)
		ttk.Entry(row1, width=6, textvariable=self.var_theta_step).grid(row=0, column=4)
		# Frequencies
		ttk.Label(section_bank, text="Frequency: start, stop, step").grid(row=2, column=0, sticky="w")
		self.var_freq_start = tk.StringVar(value="0.1")
		self.var_freq_stop = tk.StringVar(value="0.3")
		self.var_freq_step = tk.StringVar(value="0.1")
		row2 = ttk.Frame(section_bank); row2.grid(row=3, column=0, sticky="w", pady=(2, 6))
		ttk.Entry(row2, width=6, textvariable=self.var_freq_start).grid(row=0, column=0)
		ttk.Label(row2, text="to").grid(row=0, column=1, padx=4)
		ttk.Entry(row2, width=6, textvariable=self.var_freq_stop).grid(row=0, column=2)
		ttk.Label(row2, text="step").grid(row=0, column=3, padx=4)
		ttk.Entry(row2, width=6, textvariable=self.var_freq_step).grid(row=0, column=4)

		ttk.Button(section_bank, text="Create Filter Bank", command=self.on_create_bank).grid(row=4, column=0, sticky="ew", pady=(6, 0))
		self.lbl_bank_count = ttk.Label(section_bank, text="Filters: 0")
		self.lbl_bank_count.grid(row=5, column=0, sticky="w", pady=(4, 0))

		section_run = ttk.LabelFrame(self.left, text="Run", padding=8)
		section_run.grid(row=2, column=0, sticky="ew", pady=(0, 10))
		self.btn_apply = ttk.Button(section_run, text="Apply Filters to Image", command=self.on_apply_filters, state="disabled")
		self.btn_apply.grid(row=0, column=0, sticky="ew")

		section_select = ttk.LabelFrame(self.left, text="Select & Save", padding=8)
		section_select.grid(row=3, column=0, sticky="nsew")
		ttk.Label(section_select, text="Filter responses (multi-select):").grid(row=0, column=0, sticky="w")
		self.listbox = tk.Listbox(section_select, height=12, selectmode="extended", exportselection=False)
		self.listbox.grid(row=1, column=0, sticky="nsew", pady=(4, 6))
		section_select.rowconfigure(1, weight=1)
		section_select.columnconfigure(0, weight=1)
		self.btn_preview = ttk.Button(section_select, text="Preview Selected", command=self.on_preview_selected)
		self.btn_preview.grid(row=2, column=0, sticky="ew")
		self.btn_save = ttk.Button(section_select, text="Save Selected...", command=self.on_save_selected, state="disabled")
		self.btn_save.grid(row=3, column=0, sticky="ew", pady=(6, 0))

		# ---- Right visualization
		right_top = ttk.Frame(self.right)
		right_top.grid(row=0, column=0, sticky="ew")
		right_top.columnconfigure(0, weight=1)
		self.lbl_status = ttk.Label(right_top, text="Load an image to begin.")
		self.lbl_status.grid(row=0, column=0, sticky="w")

		self.fig = Figure(figsize=(7, 6), dpi=100)
		self.ax_main = self.fig.add_subplot(111)
		self.ax_main.set_axis_off()
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
		self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(6, 0))

	def _bind_events(self) -> None:
		self.listbox.bind("<<ListboxSelect>>", lambda e: self.on_listbox_select())

	# ---------------------------
	# Actions
	# ---------------------------
	def on_load_image(self) -> None:
		filepath = filedialog.askopenfilename(
			title="Select image",
			filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"), ("All files", "*.*")]
		)
		if not filepath:
			return
		path = Path(filepath)
		if path.suffix.lower() not in SUPPORTED_EXTS:
			messagebox.showerror("Unsupported format", f"Unsupported file extension: {path.suffix}")
			return
		try:
			size = (int(self.var_h.get()), int(self.var_w.get()))
		except ValueError:
			messagebox.showerror("Invalid Size", "Resize values must be integers, e.g., 256 x 256")
			return
		try:
			img = load_gray_resized(path, size=size)
		except Exception as e:
			messagebox.showerror("Load Error", f"Failed to load image:\n{e}")
			return
		self.loaded_image_path = path
		self.original_image = img
		self.target_size = size
		self._show_image(img, title=f"Original (resized {size[0]}x{size[1]})")
		self.lbl_status.config(text=f"Loaded: {path.name}")
		self.btn_apply.config(state="normal" if len(self.filter_bank) > 0 else "disabled")

	def on_apply_resize(self) -> None:
		if self.original_image is None or self.loaded_image_path is None:
			messagebox.showinfo("Info", "Load an image first.")
			return
		try:
			size = (int(self.var_h.get()), int(self.var_w.get()))
		except ValueError:
			messagebox.showerror("Invalid Size", "Resize values must be integers, e.g., 256 x 256")
			return
		try:
			img = load_gray_resized(self.loaded_image_path, size=size)
		except Exception as e:
			messagebox.showerror("Resize Error", f"Failed to resize image:\n{e}")
			return
		self.original_image = img
		self.target_size = size
		self._show_image(img, title=f"Original (resized {size[0]}x{size[1]})")
		self.lbl_status.config(text=f"Resized to {size[0]}x{size[1]}")

	def on_create_bank(self) -> None:
		try:
			t_start = float(self.var_theta_start.get())
			t_stop = float(self.var_theta_stop.get())
			t_step = float(self.var_theta_step.get())
			f_start = float(self.var_freq_start.get())
			f_stop = float(self.var_freq_stop.get())
			f_step = float(self.var_freq_step.get())
		except ValueError:
			messagebox.showerror("Invalid Parameters", "Please enter numeric values for ranges and steps.")
			return
		if t_step <= 0 or f_step <= 0:
			messagebox.showerror("Invalid Step", "Step sizes must be positive.")
			return
		# build ranges (inclusive of stop if aligns with steps)
		thetas = []
		val = t_start
		while val <= t_stop + 1e-9:
			thetas.append(round(val, 6))
			val += t_step
		freqs = []
		val = f_start
		while val <= f_stop + 1e-9:
			freqs.append(round(val, 6))
			val += f_step
		# create params
		self.filter_bank = [GaborParams(theta_deg=t, frequency=f) for t in thetas for f in freqs]
		self.lbl_bank_count.config(text=f"Filters: {len(self.filter_bank)}")
		self.listbox.delete(0, tk.END)
		for p in self.filter_bank:
			self.listbox.insert(tk.END, p.key())
		self.responses.clear()
		self.btn_apply.config(state="normal" if self.original_image is not None else "disabled")
		self.btn_save.config(state="disabled")
		self.lbl_status.config(text=f"Created bank with {len(self.filter_bank)} filters.")

	def on_apply_filters(self) -> None:
		if self.original_image is None:
			messagebox.showinfo("Info", "Load an image first.")
			return
		if not self.filter_bank:
			messagebox.showinfo("Info", "Create a filter bank first.")
			return
		self.lbl_status.config(text="Applying filters (this may take a moment)...")
		self.update_idletasks()
		img = self.original_image
		self.responses.clear()
		for p in self.filter_bank:
			try:
				mag = gabor_magnitude(img, theta_deg=p.theta_deg, frequency=p.frequency)
			except Exception as e:
				messagebox.showerror("Gabor Error", f"Error for {p.key()}:\n{e}")
				return
			self.responses[p] = mag
		self.lbl_status.config(text=f"Computed {len(self.responses)} responses.")
		self.btn_save.config(state="normal")
		# Auto show first response if any
		if len(self.responses) > 0:
			first = next(iter(self.responses.keys()))
			self._show_image(self.responses[first], title=f"Response: {first.key()}")

	def on_listbox_select(self) -> None:
		# Live preview when selecting single item
		indices = list(self.listbox.curselection())
		if len(indices) == 1 and self.responses:
			p = self.filter_bank[indices[0]]
			if p in self.responses:
				self._show_image(self.responses[p], title=f"Response: {p.key()}")

	def on_preview_selected(self) -> None:
		indices = list(self.listbox.curselection())
		if not indices:
			messagebox.showinfo("Info", "Select one or more filters in the list.")
			return
		if len(indices) == 1:
			p = self.filter_bank[indices[0]]
			arr = self.responses.get(p, None)
			if arr is None:
				messagebox.showinfo("Info", "Apply filters first.")
				return
			self._show_image(arr, title=f"Response: {p.key()}")
		else:
			# For multiple, show the first and mention count
			p = self.filter_bank[indices[0]]
			arr = self.responses.get(p, None)
			if arr is None:
				messagebox.showinfo("Info", "Apply filters first.")
				return
			self._show_image(arr, title=f"{len(indices)} responses selected (showing first: {p.key()})")

	def on_save_selected(self) -> None:
		indices = list(self.listbox.curselection())
		if not indices:
			messagebox.showinfo("Info", "Select one or more filters in the list.")
			return
		if not self.responses:
			messagebox.showinfo("Info", "Apply filters first.")
			return
		outdir = filedialog.askdirectory(title="Choose output directory")
		if not outdir:
			return
		out_path = Path(outdir)
		out_path.mkdir(parents=True, exist_ok=True)
		# Save images and JSON of parameters
		selected_params: List[GaborParams] = []
		base = self.loaded_image_path.stem if self.loaded_image_path else "image"
		for idx in indices:
			p = self.filter_bank[idx]
			arr = self.responses.get(p, None)
			if arr is None:
				continue
			selected_params.append(p)
			disp = to_display_image(arr)
			fn = f"{base}_theta{p.theta_deg:g}_freq{p.frequency:g}.png"
			try:
				io.imsave((out_path / fn).as_posix(), disp)
			except Exception as e:
				messagebox.showerror("Save Error", f"Failed to save {fn}:\n{e}")
				return
		# Save JSON manifest
		manifest = {
			"source_image": self.loaded_image_path.as_posix() if self.loaded_image_path else None,
			"resize_hw": [int(self.target_size[0]), int(self.target_size[1])],
			"selected_filters": [
				{"theta_deg": float(p.theta_deg), "frequency": float(p.frequency)} for p in selected_params
			],
		}
		try:
			with (out_path / f"{base}_selected_filters.json").open("w", encoding="utf-8") as f:
				json.dump(manifest, f, indent=2)
		except Exception as e:
			messagebox.showerror("Save Error", f"Failed to save JSON manifest:\n{e}")
			return
		messagebox.showinfo("Saved", f"Saved {len(selected_params)} images and manifest to:\n{out_path}")

	def _show_image(self, img: np.ndarray, title: str = "") -> None:
		self.ax_main.clear()
		self.ax_main.imshow(img, cmap="gray")
		self.ax_main.set_axis_off()
		self.ax_main.set_title(title)
		self.canvas.draw_idle()


def main() -> None:
	app = GaborGUI()
	app.mainloop()


if __name__ == "__main__":
	main()



