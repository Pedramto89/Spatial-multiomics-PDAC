# Notebook1: Multimodal Graph Neural Networks for Spatial Transcriptomics Analysis

A deep learning pipeline that integrates histology images and gene expression data to model disease progression from normal pancreas through primary tumors to metastatic sites.

HEAD
## Contents
- `Spatial_multimodal.ipynb`: Jupyter notebook containing the analysis pipeline and models
- The dataset utilized in this project originates from the article "Spatial transcriptomic analysis of primary and metastatic pancreatic cancers highlights tumor microenvironmental heterogeneity," published in Nature Genetics in 2024 (Article link: https://doi.org/10.1038/s41588-024-01914-4). The original data provided by the authors was stored in an RDS format. For the purposes of this project, we converted it into an h5ad file format and made it publicly accessible through the following repository: https://doi.org/10.6084/m9.figshare.28835765.v1.
=======
## 🎯 What This Does
- Combines **spatial gene expression** (17K+ genes) with **histology images**
- Uses **graph neural networks** to capture tissue microenvironment interactions  
- Predicts **tissue types** and **disease progression stages**
- Analyzes **91K+ tissue spots** across **30 slides** from pancreatic cancer samples

## ✨ Key Features

### Data Types Analyzed
- **Normal Pancreas (NP)**: Healthy tissue baseline
- **Primary Tumors (T)**: Original cancer sites  
- **Hepatic Metastases (HM)**: Liver spread
- **Lymph Node Metastases (LNM)**: Lymphatic spread

### Technical Innovations
- **Smart coordinate scaling** for accurate image patch extraction
- **Spatial graph construction** using k-nearest neighbors
- **Multimodal fusion** of CNN image features + gene expression
- **Graph convolution layers** for tissue context modeling

## 🔬 How It Works

### Step 1: Data Preparation
- Load spatial transcriptomics data (`.h5ad` format)
- Extract spatial coordinates and histology images
- Identify tissue types and create train/validation splits

### Step 2: Graph Construction  
- Build spatial graphs for each tissue slide
- Connect nearby spots using k-nearest neighbors
- Capture tissue microenvironment relationships

### Step 3: Feature Extraction
- **Gene features**: Log-transformed expression (17,860 genes → 128D)
- **Image features**: CNN on histology patches (224×224 → 128D)
- **Spatial features**: Coordinate-based graph structure

### Step 4: Model Architecture
- Graph Neural Network with multimodal inputs
- Dual prediction heads: tissue classification + disease stage
- End-to-end training with spatial awareness

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for full dataset

### Installation
```bash
git clone https://github.com/yourusername/spatial-transcriptomics-gnn
cd spatial-transcriptomics-gnn
pip install -r requirements.txt
```

### Basic Usage
```python
# 1. Data preprocessing
jupyter notebook notebooks/01_data_preprocessing_and_setup.ipynb

# 2. Model training  
jupyter notebook notebooks/02_model_training.ipynb
```

## 📊 Key Results

### Dataset Statistics
- **Total spots**: 91,496 tissue locations
- **Slides analyzed**: 30 (pancreatic cancer progression)
- **Feature extraction**: 100% success rate
- **Training split**: 80% train / 20% validation

### Model Performance
- **Tissue classification**: [Add your accuracy here]
- **Disease stage prediction**: [Add your results here]
- **Spatial coherence**: Graph structure captures tissue organization

### Improvements Achieved
- **Image extraction**: 8.1% → 100% success rate (12x improvement)
- **Processing speed**: 10-20x faster with pre-extracted features
- **Model quality**: Multimodal fusion vs single-modality approaches

## 💡 Technical Highlights

### Graph Construction Strategy
- **K-nearest neighbors**: k=8 connections per spot
- **Coordinate scaling**: Fixed scaling issues for 100% patch extraction
- **Graph normalization**: Symmetric normalization for stable training

### Image Processing Pipeline
- **Patch size**: 224×224 pixels per tissue spot
- **CNN architecture**: Conv2D → MaxPool → GlobalAvgPool
- **Feature dimension**: 128D per spot
- **Success rate**: 100% (improved from initial 8.1%)

### Model Architecture
```python
# Simplified model structure
Gene Expression (17,860) → Dense(128) 
Image Patches (224×224×3) → CNN → (128)
Combined Features (256) → GCN Layers → Predictions
```

## 📋 Data Requirements
- Spatial transcriptomics data in `.h5ad` format
- High-resolution histology images
- Spatial coordinate mappings
- Tissue type annotations

## ⚠️ Important Notes
- **Memory requirements**: ~6GB for gene expression data
- **Processing time**: ~37 minutes for image feature extraction
- **GPU recommended**: For CNN feature extraction
- **Checkpoint system**: Saves all processed data for faster restarts

## 📦 Dependencies

### Core Libraries
```
squidpy>=1.2.3
scanpy>=1.9.8
tensorflow>=2.12.0
numpy>=1.22.4
pandas>=2.0.3
scikit-learn>=0.24.2
matplotlib>=3.7.5
h5py>=3.11.0
```

### Optional (for enhanced performance)
```
umap-learn>=0.5.7
scipy>=1.10.1
```

## 🔄 Workflow Overview

1. **Data Loading & Inspection** → Load `.h5ad` files, check spatial coordinates
2. **Sample Type Assignment** → Classify spots by tissue type (NP/T/HM/LNM)
3. **Graph Construction** → Build spatial graphs for each slide
4. **Image Feature Extraction** → Process histology patches with CNN
5. **Model Training** → Train multimodal GNN
6. **Evaluation** → Assess tissue classification and disease stage prediction

## 🚧 Current Status
- ✅ Data preprocessing pipeline complete
- ✅ Graph construction implemented
- ✅ Image feature extraction optimized (100% success)
- ✅ Model architecture defined
- 🔄 Training and evaluation (in progress;can be found on Notebook #2)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
12cbdbd0183f9919806c1db8eee9e043e98bc241
