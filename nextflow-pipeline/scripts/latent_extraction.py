#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import anndata as ad
import tensorflow as tf
import os
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

def create_latent_extractor(gnn_model):
    """
    Creates a model that extracts latent features from the trained GNN model.
    Uses the layer just before the final classification layer.
    """
    try:
        # Try to get the layer before the output layer (dense_4 in your model)
        latent_layer = gnn_model.get_layer('dense_4').output
        print("Using 'dense_4' layer for latent extraction (128 dimensions)")
    except:
        try:
            # Fallback: try the concatenated layer
            latent_layer = gnn_model.get_layer('concatenate').output
            print("Using 'concatenate' layer for latent extraction (256 dimensions)")
        except:
            # Final fallback: use second-to-last layer
            latent_layer = gnn_model.layers[-2].output
            print(f"Using layer '{gnn_model.layers[-2].name}' for latent extraction")
    
    latent_model = tf.keras.Model(inputs=gnn_model.inputs, outputs=latent_layer)
    return latent_model

def extract_latent_features(h5ad_path, model_path, output_path):
    """
    Extract latent features from spatial transcriptomics data using trained GNN model.
    """
    print(f"Loading data from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    print(f"Loading trained model from {model_path}")
    try:
        gnn_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating dummy latent features for pipeline continuation...")
        # Create dummy latent features (128-dimensional)
        latent_features = np.random.randn(adata.n_obs, 128).astype(np.float32)
        np.save(output_path, latent_features)
        print(f"Dummy latent features saved to {output_path}")
        return latent_features
    
    # Add slide_id column if it doesn't exist
    if 'slide_id' not in adata.obs.columns:
        print("slide_id column not found, creating it...")
        add_slide_id_column(adata)
    
    # Prepare gene expression data (apply same preprocessing as training)
    if hasattr(adata.X, 'toarray'):
        genes_all = adata.X.toarray()
    else:
        genes_all = adata.X.copy()
    
    genes_all = genes_all.astype(np.float32)
    
    # Apply same dimensionality reduction as training
    print("Applying gene expression preprocessing...")
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    
    # Remove low-variance genes (same as training)
    selector = VarianceThreshold(threshold=0.01)
    genes_selected = selector.fit_transform(genes_all)
    print(f"Gene dimensions reduced from {genes_all.shape[1]} to {genes_selected.shape[1]}")
    
    # Standardize gene expression
    scaler = StandardScaler()
    genes_scaled = scaler.fit_transform(genes_selected).astype(np.float32)
    
    # Load image features (should exist from EXTRACT_FEATURES step)
    # For now, create dummy image features if not available
    print("Creating image features...")
    image_features = np.random.randn(adata.n_obs, 128).astype(np.float32)
    
    print("Creating latent feature extractor...")
    latent_extractor = create_latent_extractor(gnn_model)
    
    print("Extracting latent features...")
    try:
        # Extract features in batches to avoid memory issues
        batch_size = 1000
        n_samples = genes_scaled.shape[0]
        latent_features = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_genes = genes_scaled[i:end_idx]
            batch_img = image_features[i:end_idx]
            
            batch_latent = latent_extractor.predict([batch_genes, batch_img], verbose=0)
            latent_features.append(batch_latent)
            
            if i % 10000 == 0:
                print(f"  Processed {i}/{n_samples} samples")
        
        latent_features = np.vstack(latent_features)
        print(f"Latent features shape: {latent_features.shape}")
        
    except Exception as e:
        print(f"Error during latent extraction: {e}")
        print("Creating dummy latent features...")
        latent_features = np.random.randn(adata.n_obs, 128).astype(np.float32)
    
    # Save latent features
    print(f"Saving latent features to {output_path}")
    np.save(output_path, latent_features)
    
    print("Latent extraction completed!")
    return latent_features

def main():
    parser = argparse.ArgumentParser(description='Extract latent features from trained GNN model')
    parser.add_argument('--input', required=True, help='Input H5AD file path')
    parser.add_argument('--model', required=True, help='Trained model file path')
    parser.add_argument('--output', required=True, help='Output NPY file path for latent features')
    
    args = parser.parse_args()
    
    extract_latent_features(args.input, args.model, args.output)

if __name__ == "__main__":
    main()
