import anndata as ad
import numpy as np
import tensorflow as tf
import argparse
import os
import pandas as pd
from pathlib import Path

def add_slide_id_column(adata):
    """
    Add slide_id column to adata.obs based on which slide each cell belongs to.
    """
    # Get all unique slide names from column prefixes
    slide_names = set()
    for col in adata.obs.columns:
        if '_tissue' in col:
            slide_name = col.replace('_tissue', '')
            slide_names.add(slide_name)
    
    slide_names = sorted(list(slide_names))
    print(f"Found {len(slide_names)} slides: {slide_names[:3]}{'...' if len(slide_names) > 3 else ''}")
    
    # Create slide_id column
    slide_ids = []
    for idx, row in adata.obs.iterrows():
        cell_slide = None
        for slide_name in slide_names:
            tissue_col = f"{slide_name}_tissue"
            if tissue_col in adata.obs.columns:
                # Check if this cell belongs to this slide (non-missing tissue value)
                if pd.notna(row[tissue_col]) and row[tissue_col] != -2147483648:
                    cell_slide = slide_name
                    break
        
        if cell_slide is None:
            # If no slide found, assign to first slide as fallback
            cell_slide = slide_names[0] if slide_names else 'unknown'
        
        slide_ids.append(cell_slide)
    
    adata.obs['slide_id'] = slide_ids
    print(f"Added slide_id column with {len(set(slide_ids))} unique slides")
    return slide_names

def extract_image_patch(slide_id, spot_idx, adata, patch_size=224):
    try:
        spatial_key = f'spatial_{slide_id}'
        
        # Check if spatial key exists
        if spatial_key not in adata.obsm:
            print(f"Warning: {spatial_key} not found in adata.obsm")
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # Get the actual index in the spatial coordinates for this slide
        slide_mask = adata.obs['slide_id'] == slide_id
        slide_indices = np.where(slide_mask)[0]
        
        if spot_idx >= len(slide_indices):
            print(f"Warning: spot_idx {spot_idx} out of range for slide {slide_id}")
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # Get spatial coordinates for all cells from this slide
        all_spatial_coords = adata.obsm[spatial_key]
        spot_coord = all_spatial_coords[slide_indices[spot_idx]]
        
        if np.isnan(spot_coord).any():
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # Check if image exists
        if 'spatial' not in adata.uns or slide_id not in adata.uns['spatial']:
            print(f"Warning: No image found for slide {slide_id}")
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        if 'images' not in adata.uns['spatial'][slide_id] or 'hires' not in adata.uns['spatial'][slide_id]['images']:
            print(f"Warning: No hires image found for slide {slide_id}")
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        hires_img = adata.uns['spatial'][slide_id]['images']['hires']
        if hires_img.max() > 1.0:
            hires_img = hires_img / 255.0
        
        coords = adata.obsm[spatial_key][slide_mask]
        valid_coords = coords[~np.isnan(coords).any(axis=1)]
        
        if len(valid_coords) == 0:
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        coord_min = valid_coords.min(axis=0)
        coord_max = valid_coords.max(axis=0)
        coord_range = coord_max - coord_min
        
        if coord_range[0] == 0 or coord_range[1] == 0:
            return np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        scale_x = (hires_img.shape[1] * 0.9) / coord_range[0]
        scale_y = (hires_img.shape[0] * 0.9) / coord_range[1]
        scale = min(scale_x, scale_y)
        
        x = int((spot_coord[0] - coord_min[0]) * scale)
        y = int((spot_coord[1] - coord_min[1]) * scale)
        
        half_size = patch_size // 2
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        src_y_start = max(0, y - half_size)
        src_y_end = min(hires_img.shape[0], y + half_size)
        src_x_start = max(0, x - half_size)
        src_x_end = min(hires_img.shape[1], x + half_size)
        
        dst_y_start = max(0, half_size - y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, half_size - x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        if (src_y_end > src_y_start) and (src_x_end > src_x_start):
            patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                hires_img[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return patch
        
    except Exception as e:
        print(f"Error extracting patch for slide {slide_id}, spot {spot_idx}: {e}")
        return np.zeros((patch_size, patch_size, 3), dtype=np.float32)

def create_cnn_extractor():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    return model

def extract_all_features(h5ad_path, output_path):
    print(f"Loading data from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    print(f"Data shape: {adata.shape}")
    
    # Add slide_id column if it doesn't exist
    if 'slide_id' not in adata.obs.columns:
        print("Adding slide_id column...")
        slide_names = add_slide_id_column(adata)
    else:
        print("slide_id column already exists")
    
    print("Creating CNN feature extractor...")
    cnn_extractor = create_cnn_extractor()
    
    slide_ids = adata.obs["slide_id"].values
    features = np.zeros((adata.n_obs, 128), dtype=np.float32)
    
    print(f"Extracting features for {adata.n_obs} cells...")
    
    # Process cells by slide for efficiency
    unique_slides = np.unique(slide_ids)
    processed = 0
    
    for slide_id in unique_slides:
        print(f"Processing slide: {slide_id}")
        slide_mask = slide_ids == slide_id
        slide_indices = np.where(slide_mask)[0]
        
        for j, global_idx in enumerate(slide_indices):
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{adata.n_obs} cells ({processed/adata.n_obs*100:.1f}%)")
            
            patch = extract_image_patch(slide_id, j, adata)
            patch = np.expand_dims(patch, axis=0)
            feature = cnn_extractor(patch)
            features[global_idx] = feature[0].numpy()
            processed += 1
    
    print(f"Saving features to {output_path}")
    np.save(output_path, features)
    print(f"Image features saved to {output_path}")
    print(f"Feature matrix shape: {features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file path")
    args = parser.parse_args()
    
    extract_all_features(args.input, args.output)
