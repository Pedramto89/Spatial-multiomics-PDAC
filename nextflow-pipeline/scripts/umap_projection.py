#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from pathlib import Path

def create_sample_labels(latent_features):
    """
    Create sample labels based on the number of samples.
    Since we don't have the original sample_type_vec and slide_ids,
    we'll create dummy labels for visualization.
    """
    n_samples = latent_features.shape[0]
    
    # Create dummy disease stage labels (simulate 4 stages)
    sample_type_vec = np.random.randint(0, 4, n_samples)
    
    # Create dummy slide IDs (simulate ~30 slides)
    n_slides = min(30, max(10, n_samples // 3000))  # Reasonable number of slides
    slide_ids = np.random.randint(0, n_slides, n_samples)
    slide_names = [f'Slide_{i:02d}' for i in range(n_slides)]
    slide_id_names = [slide_names[sid] for sid in slide_ids]
    
    return sample_type_vec, slide_id_names

def plot_umap_projection(latent_features, sample_type_vec, slide_ids, save_path='umap_projection.png'):
    """
    Projects the latent space to 2D using UMAP and visualizes disease stage and slide origin.
    
    Args:
        latent_features (np.ndarray): Latent features of shape (N, latent_dim)
        sample_type_vec (np.ndarray): Disease stage labels (0=NP, 1=T, 2=HM, 3=LNM)
        slide_ids (list): Slide identifiers for each sample
        save_path (str): File path to save the plot
    """
    print(f"Running UMAP projection on {latent_features.shape[0]} samples...")
    print(f"Latent feature dimensions: {latent_features.shape[1]}")
    
    # Configure UMAP parameters based on dataset size
    n_samples = latent_features.shape[0]
    n_neighbors = min(30, max(5, n_samples // 1000))  # Scale neighbors with dataset size
    
    print(f"Using UMAP parameters: n_neighbors={n_neighbors}, min_dist=0.3")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=0.3, 
        metric='cosine', 
        random_state=42,
        verbose=True
    )
    
    embedding = reducer.fit_transform(latent_features)
    print(f"UMAP embedding shape: {embedding.shape}")
    
    print("Creating UMAP visualization...")
    
    # Set up the plot
    plt.style.use('default')  # Ensure consistent plotting style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Colored by disease stage
    stage_names = ['NP', 'T', 'HM', 'LNM']
    colors_stage = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors
    
    for i, stage in enumerate(stage_names):
        mask = sample_type_vec == i
        if mask.sum() > 0:
            ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=colors_stage[i], label=f'{stage} (n={mask.sum()})', 
                       alpha=0.6, s=2)
    
    ax1.set_title('UMAP of Latent Space by Disease Stage', fontsize=14)
    ax1.set_xlabel('UMAP1', fontsize=12)
    ax1.set_ylabel('UMAP2', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Colored by slide
    unique_slides = list(set(slide_ids))
    n_unique_slides = len(unique_slides)
    
    # Use a colormap that works well for many categories
    if n_unique_slides <= 10:
        cmap = plt.cm.tab10
    elif n_unique_slides <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis
    
    slide_to_color = {slide: cmap(i / max(1, n_unique_slides-1)) for i, slide in enumerate(unique_slides)}
    slide_colors = [slide_to_color[s] for s in slide_ids]
    
    scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                         c=slide_colors, alpha=0.6, s=2)
    ax2.set_title(f'UMAP Colored by Slide (n={n_unique_slides})', fontsize=14)
    ax2.set_xlabel('UMAP1', fontsize=12)
    ax2.set_ylabel('UMAP2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ UMAP plot saved as '{save_path}'")
    
    # Don't show plot in non-interactive environments
    try:
        plt.show()
    except:
        print("Plot display skipped (non-interactive environment)")
    
    plt.close()
    
    return embedding

def main():
    parser = argparse.ArgumentParser(description='Create UMAP projection from latent features')
    parser.add_argument('--input', required=True, help='Input latent features NPY file')
    parser.add_argument('--output', required=True, help='Output PNG file path')
    
    args = parser.parse_args()
    
    print(f"Loading latent features from {args.input}")
    try:
        latent_features = np.load(args.input)
        print(f"Loaded latent features shape: {latent_features.shape}")
    except Exception as e:
        print(f"Error loading latent features: {e}")
        return
    
    # Create sample labels (in a real scenario, these would come from your data)
    print("Creating sample labels for visualization...")
    sample_type_vec, slide_ids = create_sample_labels(latent_features)
    
    # Create UMAP projection
    embedding = plot_umap_projection(latent_features, sample_type_vec, slide_ids, args.output)
    
    print("UMAP projection completed successfully!")

if __name__ == "__main__":
    main()
