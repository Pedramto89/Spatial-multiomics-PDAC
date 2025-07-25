#!/usr/bin/env python3

import anndata as ad
import numpy as np
import scipy.sparse as sp
import pickle
import argparse
from sklearn.neighbors import kneighbors_graph
from pathlib import Path

def build_spatial_graph(coords, k=6, include_self=False):
    """Build k-nearest neighbor graph from spatial coordinates."""
    valid_mask = ~np.isnan(coords).any(axis=1)
    valid_coords = coords[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_coords) < k + 1:
        return sp.csr_matrix((len(coords), len(coords))), valid_mask
    
    connectivity = kneighbors_graph(valid_coords, n_neighbors=k, include_self=include_self, mode='connectivity')
    connectivity = (connectivity + connectivity.T) > 0
    
    n_spots = len(coords)
    adj_matrix = sp.csr_matrix((n_spots, n_spots), dtype=np.float32)
    
    for i, src_idx in enumerate(valid_indices):
        for j, dst_idx in enumerate(valid_indices):
            if connectivity[i, j]:
                adj_matrix[src_idx, dst_idx] = 1
    
    return adj_matrix, valid_mask

def construct_graphs(input_path, output_path, k_neighbors=8):
    """Construct spatial graphs for all slides in the dataset."""
    print("Loading input AnnData...")
    adata = ad.read_h5ad(input_path)
    
    slide_graphs = {}
    valid_masks = {}
    
    # Get spatial entries from adata.uns
    spatial_entries = adata.uns.get("spatial", {})
    print(f"Found {len(spatial_entries)} spatial entries...")
    
    for slide in spatial_entries:
        print(f"Processing slide {slide}...")
        spatial_key = f"spatial_{slide}"
        
        if spatial_key not in adata.obsm:
            print(f"  Skipping {slide}: No spatial key {spatial_key} found.")
            continue
        
        coords = adata.obsm[spatial_key]
        adj_matrix, valid_mask = build_spatial_graph(coords, k=k_neighbors)
        
        slide_graphs[slide] = adj_matrix
        valid_masks[slide] = valid_mask
        
        print(f"  - {valid_mask.sum()} valid spots, {adj_matrix.sum() // 2} edges")
    
    # Save outputs
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "slide_graphs.pkl", "wb") as f:
        pickle.dump(slide_graphs, f)
    
    with open(output_dir / "valid_masks.pkl", "wb") as f:
        pickle.dump(valid_masks, f)
    
    print(f"Graphs and masks saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct spatial graphs for slides in an AnnData object")
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save graph and mask pickle files")
    parser.add_argument("--k", type=int, default=8, help="Number of neighbors for k-NN graph")
    
    args = parser.parse_args()
    construct_graphs(args.input, args.output, args.k)
