#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from scipy import sparse

def simple_data_generator(indices, genes_all, image_features, labels, batch_size=32, shuffle=True):
    """
    Simple data generator for the current model architecture (no adjacency matrix).
    
    Args:
        indices: Array of indices to use (training or validation)
        genes_all: Gene expression array 
        image_features: Pre-extracted image features array
        labels: One-hot encoded labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle data each epoch
        
    Yields:
        ([gene_batch, img_batch], label_batch): Input and label batches
    """
    n_samples = len(indices)
    
    print(f"Simple data generator initialized:")
    print(f"  - {n_samples} total samples")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Shuffle: {shuffle}")
    
    while True:
        # Shuffle indices each epoch
        if shuffle:
            epoch_indices = np.random.permutation(indices)
        else:
            epoch_indices = indices.copy()
        
        # Generate batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_indices = epoch_indices[i:end_idx]
            
            # Skip very small batches
            if len(batch_indices) < 2:
                continue
            
            # Extract batch data
            batch_genes = genes_all[batch_indices]
            batch_imgs = image_features[batch_indices]
            batch_labels = labels[batch_indices]
            
            yield [batch_genes, batch_imgs], batch_labels

def graph_aware_data_generator(indices, genes_all, image_features, labels, 
                              slide_ids, slide_graphs, batch_size=32, shuffle=True):
    """
    Graph-aware data generator that uses spatial adjacency matrices.
    This version groups samples by slide and uses graph structure.
    
    Args:
        indices: Array of indices to use
        genes_all: Gene expression array
        image_features: Pre-extracted image features
        labels: One-hot encoded labels
        slide_ids: Array of slide IDs for each sample
        slide_graphs: Dictionary of spatial graphs (slide_id -> adjacency matrix)
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Yields:
        ([gene_batch, img_batch, adj_batch], label_batch): Input and label batches
    """
    # Group indices by slide
    slide_to_indices = {}
    for idx in indices:
        slide_id = slide_ids[idx]
        if slide_id not in slide_to_indices:
            slide_to_indices[slide_id] = []
        slide_to_indices[slide_id].append(idx)
    
    print(f"Graph-aware data generator initialized:")
    print(f"  - {len(indices)} total samples")
    print(f"  - {len(slide_to_indices)} slides")
    print(f"  - Batch size: {batch_size}")
    
    while True:
        slides = list(slide_to_indices.keys())
        if shuffle:
            np.random.shuffle(slides)
        
        for slide_id in slides:
            slide_indices = slide_to_indices[slide_id]
            
            # Skip slides with too few samples
            if len(slide_indices) < 2:
                continue
            
            # Get adjacency matrix for this slide
            if slide_id not in slide_graphs:
                print(f"Warning: No graph found for slide {slide_id}")
                continue
                
            adj_matrix = slide_graphs[slide_id]
            
            # Convert to dense if sparse
            if sparse.issparse(adj_matrix):
                adj_matrix = adj_matrix.tocsr()
                sub_adj = adj_matrix[slide_indices, :][:, slide_indices]
                adj_dense = sub_adj.toarray().astype(np.float32)
            else:
                adj_dense = adj_matrix[slide_indices][:, slide_indices].astype(np.float32)
            
            # Normalize adjacency matrix (simple row normalization)
            row_sum = np.sum(adj_dense, axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1  # Avoid division by zero
            normalized_adj = adj_dense / row_sum
            
            # Extract features for this slide
            gene_data = genes_all[slide_indices]
            img_data = image_features[slide_indices]
            label_data = labels[slide_indices]
            
            n_samples = len(gene_data)
            
            # Shuffle within slide if requested
            if shuffle:
                rand_indices = np.random.permutation(n_samples)
            else:
                rand_indices = np.arange(n_samples)
            
            # Generate batches for this slide
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                
                # Skip very small batches
                if end_idx - i < 2:
                    continue
                
                batch_indices = rand_indices[i:end_idx]
                
                # Extract batch data
                batch_genes = gene_data[batch_indices]
                batch_imgs = img_data[batch_indices]
                batch_adj = normalized_adj[batch_indices][:, batch_indices]
                batch_labels = label_data[batch_indices]
                
                yield [batch_genes, batch_imgs, batch_adj], batch_labels

def create_tensorflow_dataset(indices, genes_all, image_features, labels, 
                            batch_size=32, shuffle=True, prefetch_buffer=tf.data.AUTOTUNE):
    """
    Create a TensorFlow dataset for more efficient training.
    
    Args:
        indices: Array of indices to use
        genes_all: Gene expression array
        image_features: Image features array
        labels: Labels array
        batch_size: Batch size
        shuffle: Whether to shuffle
        prefetch_buffer: Prefetch buffer size
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Extract data for the specified indices
    dataset_genes = genes_all[indices]
    dataset_imgs = image_features[indices]
    dataset_labels = labels[indices]
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'genes': dataset_genes,
        'images': dataset_imgs,
        'labels': dataset_labels
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(indices))
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x: ([x['genes'], x['images']], x['labels']),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the data generators
    
    # Create dummy data for testing
    n_samples = 1000
    gene_dim = 500
    img_dim = 128
    n_classes = 4
    
    genes = np.random.randn(n_samples, gene_dim).astype(np.float32)
    images = np.random.randn(n_samples, img_dim).astype(np.float32)
    labels = tf.keras.utils.to_categorical(np.random.randint(0, n_classes, n_samples))
    
    indices = np.arange(n_samples)
    
    print("Testing simple data generator:")
    gen = simple_data_generator(indices, genes, images, labels, batch_size=32)
    
    # Test a few batches
    for i, (inputs, targets) in enumerate(gen):
        print(f"Batch {i+1}:")
        print(f"  Gene features shape: {inputs[0].shape}")
        print(f"  Image features shape: {inputs[1].shape}")
        print(f"  Labels shape: {targets.shape}")
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\nTesting TensorFlow dataset:")
    tf_dataset = create_tensorflow_dataset(indices, genes, images, labels, batch_size=32)
    
    for i, (inputs, targets) in enumerate(tf_dataset.take(3)):
        print(f"TF Batch {i+1}:")
        print(f"  Gene features shape: {inputs[0].shape}")
        print(f"  Image features shape: {inputs[1].shape}")
        print(f"  Labels shape: {targets.shape}")
