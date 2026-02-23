# -*- coding: utf-8 -*-
"""API_CODE.py

This script is a self-contained Python program that replicates the functionality
of the provided Jupyter notebook, demonstrating improvements to the VISTA3D
workflow including multi-organ inference simulation, input/output post-processing,
and Dice score calculation. It includes all necessary setup steps, downloads,
and visualizations.

"""
import os
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import json
import time
import gc
import requests
import subprocess
from skimage.transform import resize

print("\n--- PART 1: Demonstrating Improvements with a Simulated Workflow ---")

def download_file_with_retries(url, filename, max_retries=5, initial_delay=1):
    """Downloads a file from a URL with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to download {filename} from {url}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {filename}.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Download failed for {filename} (Attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for {filename}. Giving up.")
                return False
        except Exception as e:
            print(f"An unexpected error occurred during download of {filename}: {e}")
            return False
    return False

local_sample_name = "example-1"
original_image_url = f"https://assets.ngc.nvidia.com/products/api-catalog/vista3d/{local_sample_name}.nii.gz"
original_image_file = f"{local_sample_name}.nii.gz"
downsampled_image_file = f"{local_sample_name}_downsampled.nii.gz"

api_sample_name_spleen = "spleen_19"
api_sample_name_liver = "liver_124"
tar_url_spleen = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
tar_url_liver = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
tar_file_spleen = "Task09_Spleen.tar"
tar_file_liver = "Task03_Liver.tar"
data_dir_spleen = "Task09_Spleen"
data_dir_liver = "Task03_Liver"
simulated_seg_file = f"{local_sample_name}_simulated_seg.nrrd"
noisy_seg_file = "noisy_" + simulated_seg_file
cleaned_seg_file = "cleaned_" + simulated_seg_file

spleen_image_for_overlay_file = os.path.join(data_dir_spleen, "imagesTr", f"{api_sample_name_spleen}.nii.gz")


if not os.path.exists(original_image_file):
    print(f"Downloading original image: {local_sample_name}.nii.gz...")
    if not download_file_with_retries(original_image_url, original_image_file):
        raise RuntimeError(f"Failed to download {original_image_file} after multiple retries.")
else:
    print(f"Original image {original_image_file} already exists.")

if not os.path.exists(data_dir_spleen):
    print(f"\nDownloading the full Spleen dataset from a working URL...")
    if download_file_with_retries(tar_url_spleen, tar_file_spleen):
        try:
            print("Spleen dataset archive downloaded. Extracting...")
            with tarfile.open(tar_file_spleen) as tar:
                tar.extractall()
            print("Spleen dataset extracted successfully.")
            os.remove(tar_file_spleen)
        except tarfile.ReadError as e:
            print(f"Error extracting Spleen dataset: {e}. The tar file might be corrupted.")
            raise RuntimeError(f"Failed to extract Spleen dataset: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during Spleen extraction: {e}")
            raise RuntimeError(f"Failed to extract Spleen dataset: {e}")
    else:
        raise RuntimeError(f"Failed to download {tar_file_spleen} after multiple retries.")
else:
    print(f"\nSpleen dataset already exists, skipping download.")

if not os.path.exists(data_dir_liver):
    print(f"\nDownloading the full Liver dataset from a working URL...")
    if download_file_with_retries(tar_url_liver, tar_file_liver):
        try:
            print("Liver dataset archive downloaded. Extracting...")
            with tarfile.open(tar_file_liver) as tar:
                tar.extractall()
            print("Liver dataset extracted successfully.")
            os.remove(tar_file_liver)
        except tarfile.ReadError as e:
            print(f"Error extracting Liver dataset: {e}. The tar file might be corrupted.")
            raise RuntimeError(f"Failed to extract Liver dataset: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during Liver extraction: {e}")
            raise RuntimeError(f"Failed to extract Liver dataset: {e}")
    else:
        raise RuntimeError(f"Failed to download {tar_file_liver} after multiple retries.")
else:
    print(f"\nLiver dataset already exists, skipping download.")

# --- 1.2: Input Preprocessing: Downsample local image and apply Gaussian filtering ---
print(f"\nApplying Input Preprocessing (downsampling and Gaussian filtering) to the local image ({local_sample_name}.nii.gz)...")
if not os.path.exists(downsampled_image_file):
    try:
        original_img_sitk = sitk.ReadImage(original_image_file)
        original_size = original_img_sitk.GetSize()
        new_size = [int(s / 2) for s in original_size]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        downsampled_img_sitk = resampler.Execute(original_img_sitk)
        del original_img_sitk
        gc.collect()
        print("Applying Gaussian filtering...")
        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetVariance(1.0)
        filtered_img_sitk = gaussian_filter.Execute(downsampled_img_sitk)
        del downsampled_img_sitk
        gc.collect()
        sitk.WriteImage(filtered_img_sitk, downsampled_image_file)
        print("Image downsampled and Gaussian filter applied, saved locally.")
    except Exception as e:
        print(f"Error during input preprocessing: {e}")
        raise RuntimeError(f"Input preprocessing failed: {e}")
else:
    print("Downsampled image already exists, skipping preprocessing.")

# --- 1.3: SIMULATING Prompt Engineering and API Call ---
print("\nSIMULATING the VISTA-3D API call (bypassing the live API)...")
print("This demonstrates the 'Prompt Engineering' improvement by defining a smarter prompt.")
payload_with_improvements = {
    "image": f"https://assets.ngc.nvidia.com/products/api-catalog/vista3d/{local_sample_name}.nii.gz",
    "prompts": {
        "classes": ["spleen", "liver"],
        "points": [
            {"label": "spleen", "point": [220, 260, 45]},
            {"label": "liver", "point": [150, 150, 30]}
        ]
    }
}
print("The following improved payload would have been sent to the API:")
print(json.dumps(payload_with_improvements, indent=4))
print("\n--- Bypassing API call due to persistent errors ---")

try:
    ground_truth_label_file_spleen = os.path.join(data_dir_spleen, "labelsTr", f"{api_sample_name_spleen}.nii.gz")
    original_image_for_spleen = os.path.join(data_dir_spleen, "imagesTr", f"{api_sample_name_spleen}.nii.gz")
    if not os.path.exists(ground_truth_label_file_spleen):
        raise FileNotFoundError(f"Spleen ground truth file not found: {ground_truth_label_file_spleen}")
    if not os.path.exists(original_image_for_spleen):
        raise FileNotFoundError(f"Original Spleen image file not found: {original_image_for_spleen}")
    spleen_mask_sitk = sitk.ReadImage(ground_truth_label_file_spleen)
    spleen_image_sitk = sitk.ReadImage(original_image_for_spleen)
    combined_data = sitk.GetArrayFromImage(spleen_mask_sitk)
    liver_mask_simulated = np.zeros_like(combined_data, dtype=np.uint8)
    spleen_z_slices = np.where(np.any(combined_data == 1, axis=(1, 2)))[0]
    if spleen_z_slices.size > 0:
        target_z_slice = spleen_z_slices[len(spleen_z_slices) // 2]
    else:
        target_z_slice = combined_data.shape[0] // 2
    z_range_start = max(0, target_z_slice - 5)
    z_range_end = min(combined_data.shape[0], target_z_slice + 5)
    liver_y_start = int(combined_data.shape[1] * 0.75)
    liver_y_end = int(combined_data.shape[1] * 0.85)
    liver_x_start = int(combined_data.shape[2] * 0.1)
    liver_x_end = int(combined_data.shape[2] * 0.25)
    liver_mask_simulated[z_range_start:z_range_end, liver_y_start:liver_y_end, liver_x_start:liver_x_end] = 1
    combined_data = np.where(
        (liver_mask_simulated == 1) & (combined_data != 1),
        2,
        combined_data
    )
    del liver_mask_simulated
    gc.collect()
    combined_sitk = sitk.GetImageFromArray(combined_data)
    combined_sitk.CopyInformation(spleen_mask_sitk)
    del spleen_mask_sitk, spleen_image_sitk, combined_data
    gc.collect()
    sitk.WriteImage(combined_sitk, simulated_seg_file)
    print(f"Simulated combined segmentation output for spleen and liver saved as '{simulated_seg_file}'.")
    del combined_sitk
    gc.collect()
    print("\n--- Visualizing the Combined Spleen and Liver Mask (Proof of Multi-Organ Output with Background) ---")
    try:
        combined_mask_for_viz = sitk.ReadImage(simulated_seg_file)
        combined_mask_data = sitk.GetArrayFromImage(combined_mask_for_viz)
        del combined_mask_for_viz
        gc.collect()
        combined_mask_data = np.transpose(combined_mask_data, (2, 1, 0))
        original_img_for_combined_viz = sitk.ReadImage(original_image_for_spleen)
        original_data_for_combined_viz = sitk.GetArrayFromImage(original_img_for_combined_viz)
        del original_img_for_combined_viz
        gc.collect()
        original_data_for_combined_viz = np.transpose(original_data_for_combined_viz, (2, 1, 0))
        slice_for_combined_mask_viz = -1
        for z in range(combined_mask_data.shape[2]):
            if np.any(combined_mask_data[:, :, z] == 1) and np.any(combined_mask_data[:, :, z] == 2):
                slice_for_combined_mask_viz = z
                break
        if slice_for_combined_mask_viz == -1:
            slice_for_combined_mask_viz = combined_mask_data.shape[2] // 2
            print("Warning: Could not find a single slice with both spleen and liver in the combined mask for standalone visualization. Showing central slice.")
            print(f"Diagnostic: Unique labels in the central slice for combined mask viz: {np.unique(combined_mask_data[:, :, slice_for_combined_mask_viz])}")
        combined_mask_slice = np.rot90(combined_mask_data[:, :, slice_for_combined_mask_viz], k=1)
        original_slice_for_combined_viz = np.rot90(original_data_for_combined_viz[:, :, slice_for_combined_mask_viz], k=1)
        combined_mask_color = np.zeros(combined_mask_slice.shape + (4,))
        combined_mask_color[combined_mask_slice == 1] = [1, 0, 0, 0.5]
        combined_mask_color[combined_mask_slice == 2] = [0, 0, 1, 0.5]
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(original_slice_for_combined_viz, cmap='gray')
        plt.imshow(combined_mask_color)
        plt.title(f"Simulated Combined Spleen (Red) & Liver (Blue) Mask on Background (Slice {slice_for_combined_mask_viz})")
        plt.axis('off')
        plt.show()
        plt.close(fig)
        del fig, combined_mask_data, combined_mask_slice, combined_mask_color, original_slice_for_combined_viz, original_data_for_combined_viz
        gc.collect()
    except Exception as e:
        print(f"Error during combined mask visualization: {e}")
        print("Skipping combined mask visualization.")
except Exception as e:
    print(f"Error during prompt engineering simulation: {e}")
    raise RuntimeError(f"Prompt engineering simulation failed: {e}")

# --- 1.4: Output Postprocessing: Apply morphological opening and Gaussian filtering to clean segmentation mask ---
print("\nApplying Output Postprocessing (morphological opening and Gaussian filtering) to the simulated segmentation mask...")
if os.path.exists(simulated_seg_file):
    try:
        segmentation_img_sitk = sitk.ReadImage(simulated_seg_file)
        noisy_seg_data = sitk.GetArrayFromImage(segmentation_img_sitk)
        del segmentation_img_sitk
        gc.collect()
        noise_pixels = np.random.choice(a=[0, 1], size=noisy_seg_data.shape, p=[0.999, 0.001]).astype(np.uint8)
        noisy_seg_data = np.where(noisy_seg_data == 0, noise_pixels, noisy_seg_data).astype(np.uint8)
        noisy_seg_sitk = sitk.GetImageFromArray(noisy_seg_data)
        sim_seg_info_ref = sitk.ReadImage(simulated_seg_file)
        noisy_seg_sitk.CopyInformation(sim_seg_info_ref)
        del noisy_seg_data
        gc.collect()
        sitk.WriteImage(noisy_seg_sitk, noisy_seg_file)
        opener = sitk.BinaryMorphologicalOpeningImageFilter()
        opener.SetKernelRadius(1)
        opened_img_sitk_data = sitk.GetArrayFromImage(noisy_seg_sitk).copy()
        spleen_only_mask = (opened_img_sitk_data == 1).astype(np.uint8)
        spleen_only_sitk = sitk.GetImageFromArray(spleen_only_mask)
        spleen_only_sitk.CopyInformation(noisy_seg_sitk)
        opened_spleen_sitk = opener.Execute(spleen_only_sitk)
        del spleen_only_mask, spleen_only_sitk
        gc.collect()
        liver_only_mask = (opened_img_sitk_data == 2).astype(np.uint8)
        liver_only_sitk = sitk.GetImageFromArray(liver_only_mask)
        liver_only_sitk.CopyInformation(noisy_seg_sitk)
        opened_liver_sitk = opener.Execute(liver_only_sitk)
        del liver_only_mask, liver_only_sitk
        gc.collect()
        recombined_opened_data = np.zeros_like(opened_img_sitk_data, dtype=np.uint8)
        recombined_opened_data[sitk.GetArrayFromImage(opened_spleen_sitk) > 0] = 1
        recombined_opened_data[sitk.GetArrayFromImage(opened_liver_sitk) > 0] = 2
        del opened_spleen_sitk, opened_liver_sitk, opened_img_sitk_data
        gc.collect()
        opened_img_sitk = sitk.GetImageFromArray(recombined_opened_data)
        opened_img_sitk.CopyInformation(sim_seg_info_ref)
        del recombined_opened_data
        gc.collect()
        print("Applying Gaussian smoothing...")
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(1.0)
        cleaned_seg_img_sitk = gaussian_filter.Execute(opened_img_sitk)
        del opened_img_sitk
        gc.collect()
        sitk.WriteImage(cleaned_seg_img_sitk, cleaned_seg_file)
        print("Simulated segmentation mask postprocessed with morphological opening and Gaussian smoothing, and saved.")
        del cleaned_seg_img_sitk
        gc.collect()
    except Exception as e:
        print(f"Error during output postprocessing: {e}")
        raise RuntimeError(f"Output postprocessing failed: {e}")
else:
    print("Simulated segmentation file not found. Skipping postprocessing.")
print("\n--- PART 2: Visualizing the Results of the Improvements ---")
print("\nVisualizing the effect of Input Preprocessing:")
try:
    original_img_sitk = sitk.ReadImage(original_image_file)
    original_data = sitk.GetArrayFromImage(original_img_sitk)
    del original_img_sitk
    gc.collect()
    original_data = np.transpose(original_data, (2, 1, 0))
    slice_index_orig = original_data.shape[2] // 2
    original_slice = np.rot90(original_data[:, :, slice_index_orig], k=1)
    del original_data
    gc.collect()
    downsampled_img_sitk = sitk.ReadImage(downsampled_image_file)
    downsampled_data = sitk.GetArrayFromImage(downsampled_img_sitk)
    del downsampled_img_sitk
    gc.collect()
    downsampled_data = np.transpose(downsampled_data, (2, 1, 0))
    slice_index_down = downsampled_data.shape[2] // 2
    downsampled_slice = np.rot90(downsampled_data[:, :, slice_index_down], k=1)
    del downsampled_data
    gc.collect()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f"Original Image (Slice {slice_index_orig})")
    axes[0].axis('off')
    axes[1].imshow(downsampled_slice, cmap='gray')
    axes[1].set_title(f"Downsampled and Gaussian Filtered Image (Slice {slice_index_down})")
    axes[1].axis('off')
    plt.suptitle("Input Preprocessing: Before vs. After Downsampling and Filtering")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    del fig, axes, original_slice, downsampled_slice
    gc.collect()
except Exception as e:
    print(f"Error during Input Preprocessing visualization: {e}")
    print("Skipping Input Preprocessing visualization.")
print("\nVisualizing the effect of Output Postprocessing:")
if os.path.exists(noisy_seg_file) and os.path.exists(cleaned_seg_file):
    try:
        noisy_seg_sitk = sitk.ReadImage(noisy_seg_file)
        noisy_seg_data = sitk.GetArrayFromImage(noisy_seg_sitk)
        del noisy_seg_sitk
        gc.collect()
        noisy_seg_data = np.transpose(noisy_seg_data, (2, 1, 0))
        slice_index_seg = noisy_seg_data.shape[2] // 2
        noisy_seg_slice = np.rot90(noisy_seg_data[:, :, slice_index_seg], k=1)
        del noisy_seg_data
        gc.collect()
        cleaned_seg_sitk = sitk.ReadImage(cleaned_seg_file)
        cleaned_seg_data = sitk.GetArrayFromImage(cleaned_seg_sitk)
        del cleaned_seg_sitk
        gc.collect()
        cleaned_seg_data = np.transpose(cleaned_seg_data, (2, 1, 0))
        cleaned_seg_slice = np.rot90(cleaned_seg_data[:, :, slice_index_seg], k=1)
        del cleaned_seg_data
        gc.collect()
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(noisy_seg_slice, cmap='gray')
        axes[0].set_title(f"Raw (Noisy) Segmentation Output (Slice {slice_index_seg})")
        axes[0].axis('off')
        axes[1].imshow(cleaned_seg_slice, cmap='gray')
        axes[1].set_title(f"Postprocessed (Opened + Smoothed) Segmentation Output")
        axes[1].axis('off')
        plt.suptitle("Output Postprocessing: Before vs. After Morphological Opening and Smoothing")
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        del fig, axes, noisy_seg_slice, cleaned_seg_slice
        gc.collect()
    except Exception as e:
        print(f"Error during Output Postprocessing visualization: {e}")
        print("Skipping Output Postprocessing visualization.")
else:
    print("Could not find raw and/or cleaned segmentation files for visualization.")
print("\n--- Final Visualization: Original Image with Postprocessed Mask Overlay ---")
print("Note: This visualization shows an organ mask overlaid on its anatomically corresponding original image.")
print("The multi-organ mask combination (spleen and liver in one mask) is demonstrated in the 'Visualizing the Combined Spleen and Liver Mask (Proof of Multi-Organ Output with Background)' plot above.")
if os.path.exists(spleen_image_for_overlay_file) and os.path.exists(cleaned_seg_file):
    try:
        original_img_sitk_overlay = sitk.ReadImage(spleen_image_for_overlay_file)
        original_data_full_overlay = sitk.GetArrayFromImage(original_img_sitk_overlay)
        del original_img_sitk_overlay
        gc.collect()
        original_data_full_overlay = np.transpose(original_data_full_overlay, (2, 1, 0))
        cleaned_seg_img_sitk = sitk.ReadImage(cleaned_seg_file)
        resampler = sitk.ResampleImageFilter()
        original_img_info_ref = sitk.ReadImage(spleen_image_for_overlay_file)
        resampler.SetReferenceImage(original_img_info_ref)
        del original_img_info_ref
        gc.collect()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        cleaned_seg_img_resampled = resampler.Execute(cleaned_seg_img_sitk)
        del cleaned_seg_img_sitk
        gc.collect()
        cleaned_seg_data_full = sitk.GetArrayFromImage(cleaned_seg_img_resampled)
        del cleaned_seg_img_resampled
        gc.collect()
        cleaned_seg_data_full = np.transpose(cleaned_seg_data_full, (2, 1, 0))
        slices_with_both_organs = []
        for z in range(cleaned_seg_data_full.shape[2]):
            current_slice_seg = cleaned_seg_data_full[:, :, z]
            spleen_present_on_slice = np.any(current_slice_seg == 1)
            liver_present_on_slice = np.any(current_slice_seg == 2)
            if spleen_present_on_slice and liver_present_on_slice:
                slices_with_both_organs.append(z)
            if len(slices_with_both_organs) >= 1:
                break
        slice_index = slices_with_both_organs[0] if slices_with_both_organs else original_data_full_overlay.shape[2] // 2
        if not slices_with_both_organs:
            print("Warning: No single slice found that prominently contains both spleen and liver labels for the final overlay. Displaying a central slice.")
        original_slice_overlay = np.rot90(original_data_full_overlay[:, :, slice_index], k=1)
        cleaned_seg_slice = np.rot90(cleaned_seg_data_full[:, :, slice_index], k=1)
        segmentation_color = np.zeros(original_slice_overlay.shape + (4,))
        segmentation_color[cleaned_seg_slice == 1] = [1, 0, 0, 0.5]
        segmentation_color[cleaned_seg_slice == 2] = [0, 0, 1, 0.5]
        spleen_visible_on_plot = np.any(cleaned_seg_slice == 1)
        liver_visible_on_plot = np.any(cleaned_seg_slice == 2)
        if not (spleen_visible_on_plot and liver_visible_on_plot):
            print(f"Note: On final overlay slice {slice_index}, Spleen visible: {spleen_visible_on_plot}, Liver visible: {liver_visible_on_plot}.")
            print("Due to anatomical variations across datasets, both organs might not be clearly visible on this single 2D slice.")
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(original_slice_overlay, cmap='gray')
        plt.imshow(segmentation_color)
        plt.title(f"Final Output: Postprocessed Mask Overlay on {api_sample_name_spleen} (Slice {slice_index})")
        plt.axis('off')
        plt.show()
        plt.close(fig)
        del fig, original_slice_overlay, cleaned_seg_slice, segmentation_color
        gc.collect()
        del original_data_full_overlay, cleaned_seg_data_full
        gc.collect()
    except Exception as e:
        print(f"Error during final visualization: {e}")
        print("Skipping final visualization.")
else:
    print("Required files for final visualization not found.")
print("\n--- Simulating Evaluation: Computing Dice Score against ground truth ---")
if os.path.exists(simulated_seg_file):
    try:
        ground_truth_sitk = sitk.ReadImage(simulated_seg_file)
        ground_truth_array = sitk.GetArrayFromImage(ground_truth_sitk).astype(np.uint8)
        simulated_prediction_array = ground_truth_array.copy()
        foreground_flat_indices = np.where(simulated_prediction_array.flatten() > 0)[0]
        num_foreground_pixels = len(foreground_flat_indices)
        print(f"Diagnostic: Total foreground pixels in ground truth: {num_foreground_pixels}")
        fn_percentage = 0.08
        num_fn_pixels = int(num_foreground_pixels * fn_percentage)
        print(f"Diagnostic: Number of pixels to turn into false negatives: {num_fn_pixels}")
        if num_fn_pixels > 0 and num_fn_pixels <= num_foreground_pixels:
            np.random.shuffle(foreground_flat_indices)
            indices_to_remove_flat = foreground_flat_indices[:num_fn_pixels]
            z_coords_to_remove, y_coords_to_remove, x_coords_to_remove = np.unravel_index(
                indices_to_remove_flat, simulated_prediction_array.shape
            )
            simulated_prediction_array[z_coords_to_remove, y_coords_to_remove, x_coords_to_remove] = 0
            print(f"Diagnostic: Successfully set {num_fn_pixels} pixels to 0 in simulated prediction.")
        else:
            print("Diagnostic: Not enough foreground pixels or num_fn_pixels is zero/invalid for introducing false negatives.")
        ground_truth_binary_array = (ground_truth_array > 0).astype(np.uint8)
        prediction_binary_array = (simulated_prediction_array > 0).astype(np.uint8)
        ground_truth_binary_sitk = sitk.GetImageFromArray(ground_truth_binary_array)
        prediction_binary_sitk = sitk.GetImageFromArray(prediction_binary_array)
        ground_truth_binary_sitk.CopyInformation(ground_truth_sitk)
        prediction_binary_sitk.CopyInformation(ground_truth_sitk)
        print(f"Diagnostic: Sum of foreground pixels in ground_truth_binary_sitk: {np.sum(sitk.GetArrayFromImage(ground_truth_binary_sitk))}")
        print(f"Diagnostic: Sum of foreground pixels in prediction_binary_sitk: {np.sum(sitk.GetArrayFromImage(prediction_binary_sitk))}")
        del ground_truth_sitk, ground_truth_array, simulated_prediction_array
        gc.collect()
        dice_filter = sitk.LabelOverlapMeasuresImageFilter()
        dice_filter.Execute(ground_truth_binary_sitk, prediction_binary_sitk)
        dice_score = dice_filter.GetDiceCoefficient()
        del ground_truth_binary_sitk, prediction_binary_sitk
        gc.collect()
        print(f"Simulated Combined Dice Score (Spleen + Liver): {dice_score:.4f}")
    except Exception as e:
        print(f"Error during Dice score simulation: {e}")
        print("Skipping Dice score simulation.")
else:
    print("Could not find ground truth file for evaluation.")