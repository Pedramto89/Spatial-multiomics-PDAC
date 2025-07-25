# Spatial Transcriptomics GNN Pipeline

A **Nextflow DSL2 pipeline** for analyzing spatial transcriptomics data using **Graph Neural Networks (GNNs)** with integrated histological image features.

## 🔬 Overview

This pipeline integrates spatial transcriptomics data with graph neural networks to perform tissue classification and spatial pattern analysis. It combines gene expression data, spatial coordinates, and histological image features to classify tissue types in pancreatic cancer samples.

**Key Innovation**: Integration of molecular (gene expression) and morphological (image) features through graph neural networks for spatial tissue analysis.

## 🏗️ Pipeline Workflow

The pipeline consists of 6 main processes executed sequentially:

```
Input Data → Graph Construction → Image Features → GNN Training → Latent Extraction → UMAP Visualization → Analysis
```

1. **GRAPH**: Constructs spatial k-nearest neighbor graphs from tissue coordinates
2. **EXTRACT_FEATURES**: Extracts CNN-based features from histological image patches  
3. **TRAIN**: Trains a Graph Neural Network combining gene and image features
4. **LATENT**: Extracts latent space representations from the trained model
5. **UMAP**: Creates 2D visualizations of the latent space
6. **ANALYSIS**: Performs tissue classification analysis and generates insights

## 📁 Scripts & Components

### Core Analysis Scripts (`scripts/` folder)

| Script | Purpose | Input/Output |
|--------|---------|--------------|
| `construct_graphs.py` | Builds spatial graphs from coordinates | H5AD → spatial graphs (PKL) |
| `extract_image_features.py` | CNN feature extraction from image patches | H5AD → image features (NPY) |
| `gnn_training.py` | Trains GNN model with gene + image features | Multiple inputs → trained model (H5) |
| `latent_extraction.py` | Extracts latent representations | Model + H5AD → latent space (NPY) |
| `umap_projection.py` | Creates 2D UMAP visualizations | Latent features → UMAP plots (PNG) |
| `training_analysis.py` | Classification analysis and metrics | UMAP → performance metrics (JSON) |

### Supporting Components

| Script | Purpose |
|--------|---------|
| `model.py` | Neural network architectures and model definitions |
| `data_generator.py` | Data loading and batch processing utilities |

### Pipeline Infrastructure

| File | Purpose |
|------|---------|
| `main.nf` | Main Nextflow workflow definition (DSL2) |
| `envs/gnn.yml` | Complete conda environment specification |

## 🎯 Scalability & Reproducibility

### Scalability Features

**Multi-slide Processing**: Designed to handle 20-30+ tissue slides simultaneously with ~91K cells and ~18K genes per dataset.

**Memory Optimization**: Implements dimensionality reduction, batch processing, and efficient data structures to handle large spatial transcriptomics datasets.

**Parallel Execution**: Nextflow automatically parallelizes independent processes (GRAPH and EXTRACT_FEATURES) while maintaining proper dependencies.

**Resource Management**: Conda environment ensures consistent package versions and resource allocation across different computing environments.

### Reproducibility Advantages

**Workflow Management**: Nextflow DSL2 provides:
- Automatic checkpoint/resume functionality
- Process isolation and containerization support
- Detailed execution reporting and logging
- Dependency tracking and data provenance

**Environment Control**: Complete conda environment specification with 150+ pinned package versions ensures identical software environments across different systems.

**Version Control**: All scripts, configurations, and documentation are version-controlled, enabling exact reproduction of analyses.

**Documentation**: Execution logs, intermediate files, and work directories provide complete audit trails of computational steps.

## Usage

**Setup Environment**:
```bash
conda env create -f envs/gnn.yml
conda activate gnn-env
```

**Run Pipeline**:
```bash
nextflow run main.nf --input your_data.h5ad --outdir results
```

**Resume Execution**:
```bash
nextflow run main.nf --input your_data.h5ad --outdir results -resume
```

## Input & Output

**Input**: AnnData H5AD file with spatial transcriptomics data (gene expression + spatial coordinates + optional images)

**Output**: Spatial graphs, image features, trained GNN model, latent representations, UMAP visualizations, and classification metrics

## Applications

Designed for **pancreatic cancer spatial transcriptomics** analysis, distinguishing between:
- **Normal Pancreas (NP)**
- **Primary Tumor (T)** 
- **Hepatic Metastasis (HM)**
- **Lymph Node Metastasis (LNM)**

The approach is adaptable to other tissue types and spatial transcriptomics applications requiring integration of molecular and morphological features.

## 🛠️ Technical Implementation

**Framework**: Nextflow DSL2 for workflow management  
**Deep Learning**: TensorFlow/Keras for GNN implementation  
**Spatial Analysis**: Scanpy ecosystem for spatial transcriptomics  
**Visualization**: UMAP for dimensionality reduction and matplotlib for plotting  
**Environment**: Conda for reproducible dependency management

---

This pipeline demonstrates the integration of modern workflow management tools (Nextflow) with advanced machine learning approaches (GNNs) for reproducible, scalable spatial transcriptomics analysis.
