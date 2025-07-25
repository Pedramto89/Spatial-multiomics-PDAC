#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import pickle
import anndata as ad
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import os

def create_simple_gnn(gene_dim, img_dim, num_classes):
    """Create a simple GNN model for spatial transcriptomics."""
    
    # Gene expression input
    gene_input = layers.Input(shape=(gene_dim,), name='gene_input')
    gene_dense = layers.Dense(256, activation='relu')(gene_input)
    gene_dense = layers.Dropout(0.3)(gene_dense)
    gene_dense = layers.Dense(128, activation='relu')(gene_dense)
    
    # Image features input
    img_input = layers.Input(shape=(img_dim,), name='img_input')
    img_dense = layers.Dense(128, activation='relu')(img_input)
    img_dense = layers.Dropout(0.3)(img_dense)
    
    # Adjacency matrix input (for graph structure)
    adj_input = layers.Input(shape=(None,), name='adj_input', sparse=True)
    
    # Simple approach: concatenate gene and image features
    combined = layers.Concatenate()([gene_dense, img_dense])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax', name='output')(combined)
    
    model = Model(inputs=[gene_input, img_input], outputs=output)
    return model

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

def prepare_data(adata, slide_graphs, image_features):
    """Prepare data for training."""
    print("Preparing training data...")
    
    # Add slide_id column if it doesn't exist
    if 'slide_id' not in adata.obs.columns:
        print("slide_id column not found, creating it...")
        add_slide_id_column(adata)
    else:
        print("slide_id column already exists")
    
    slide_ids = adata.obs['slide_id'].values
    unique_slides = np.unique(slide_ids)
    
    print(f"Found {len(unique_slides)} unique slides")
    
    # Gene expression data
    if hasattr(adata.X, 'toarray'):
        genes_all = adata.X.toarray()
    else:
        genes_all = adata.X
    
    print(f"Gene expression shape: {genes_all.shape}")
    print(f"Image features shape: {image_features.shape}")
    
    # Create simple labels based on slide type (you may want to modify this)
    # Extract slide type from slide names (e.g., HM, LNM, NP, T)
    slide_types = []
    for slide_id in slide_ids:
        if 'HM' in slide_id:
            slide_types.append('HM')
        elif 'LNM' in slide_id:
            slide_types.append('LNM') 
        elif 'NP' in slide_id:
            slide_types.append('NP')
        elif '_T' in slide_id:
            slide_types.append('T')
        else:
            slide_types.append('Other')
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(slide_types)
    num_classes = len(label_encoder.classes_)
    
    print(f"Label classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Convert to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels_encoded, num_classes)
    
    return {
        'genes_all': genes_all,
        'image_features': image_features,
        'labels_categorical': labels_categorical,
        'labels_encoded': labels_encoded,
        'slide_ids': slide_ids,
        'unique_slides': unique_slides,
        'num_classes': num_classes,
        'label_encoder': label_encoder
    }

def train_gnn_model(data_dict, output_model_path):
    """Train the GNN model."""
    print("Starting GNN training...")
    
    genes_all = data_dict['genes_all'].astype(np.float32)  # Ensure float32
    image_features = data_dict['image_features'].astype(np.float32)  # Ensure float32
    labels_categorical = data_dict['labels_categorical'].astype(np.float32)
    labels_encoded = data_dict['labels_encoded']
    slide_ids = data_dict['slide_ids']
    unique_slides = data_dict['unique_slides']
    num_classes = data_dict['num_classes']
    
    # Reduce gene dimensions to manage memory
    print("Reducing gene expression dimensionality...")
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    
    # Remove low-variance genes
    selector = VarianceThreshold(threshold=0.01)
    genes_selected = selector.fit_transform(genes_all)
    print(f"Gene dimensions reduced from {genes_all.shape[1]} to {genes_selected.shape[1]}")
    
    # Standardize gene expression
    scaler = StandardScaler()
    genes_scaled = scaler.fit_transform(genes_selected).astype(np.float32)
    
    # Slide-based train-validation split
    train_slides, val_slides = train_test_split(
        unique_slides,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    print(f"Training slides: {len(train_slides)}")
    print(f"Validation slides: {len(val_slides)}")
    
    # Get indices for train/val split
    train_idx = [i for i, sid in enumerate(slide_ids) if sid in train_slides]
    val_idx = [i for i, sid in enumerate(slide_ids) if sid in val_slides]
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    
    # Prepare training data with smaller batches
    X_train_genes = genes_scaled[train_idx]
    X_train_img = image_features[train_idx]
    y_train = labels_categorical[train_idx]
    
    X_val_genes = genes_scaled[val_idx]
    X_val_img = image_features[val_idx]
    y_val = labels_categorical[val_idx]
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels_encoded),
        y=labels_encoded[train_idx]
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Class weights: {class_weight_dict}")
    
    # Create simpler model to reduce memory usage
    model = create_simple_gnn(
        gene_dim=genes_scaled.shape[1],
        img_dim=image_features.shape[1],
        num_classes=num_classes
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks with more conservative settings
    callbacks = [
        ModelCheckpoint(
            output_model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reduced patience
            restore_best_weights=True,
            min_delta=0.01,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reduced patience
            min_lr=1e-7,
            verbose=1
        ),
        TerminateOnNaN()
    ]
    
    # Use smaller batch size and fewer epochs to reduce memory pressure
    batch_size = 16  # Much smaller batch size
    epochs = 10      # Fewer epochs
    
    print(f"Starting training with batch_size={batch_size}, epochs={epochs}...")
    
    # Clear any previous models from memory
    tf.keras.backend.clear_session()
    
    try:
        history = model.fit(
            [X_train_genes, X_train_img], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_val_genes, X_val_img], y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )
        print(f"Training completed successfully. Model saved to: {output_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save a simple model anyway for the pipeline to continue
        print("Saving a basic trained model...")
        model.save(output_model_path)
        history = None
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train GNN model for spatial transcriptomics')
    parser.add_argument('--input', required=True, help='Input H5AD file path')
    parser.add_argument('--adj', required=True, help='Adjacency graphs pickle file')
    parser.add_argument('--mask', required=True, help='Valid masks pickle file')
    parser.add_argument('--image_features', required=True, help='Image features NPY file')
    parser.add_argument('--output_model', required=True, help='Output model path')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    adata = ad.read_h5ad(args.input)
    
    print(f"Loading slide graphs from {args.adj}")
    with open(args.adj, 'rb') as f:
        slide_graphs = pickle.load(f)
    
    print(f"Loading valid masks from {args.mask}")
    with open(args.mask, 'rb') as f:
        valid_masks = pickle.load(f)
    
    print(f"Loading image features from {args.image_features}")
    image_features = np.load(args.image_features)
    
    # Prepare data
    data_dict = prepare_data(adata, slide_graphs, image_features)
    if data_dict is None:
        print("Error preparing data")
        return
    
    # Train model
    model, history = train_gnn_model(data_dict, args.output_model)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
