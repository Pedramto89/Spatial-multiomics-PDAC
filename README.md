**Multi-Modal VAE with Modality-Specific Priors**

Overview

This repository contains an implementation of a Multi-Modal Variational Autoencoder (VAE) designed for multi-omics data fusion. The model leverages modality-specific variational priors, allowing each omics layer to have its own learned distribution, which improves biological interpretability and robustness to missing data.

Key Features

- Modality-Specific Priors

Each omics layer (e.g., Spatial, RNA, ATAC) has its own learned prior distribution. We encode the high-quality cells/spots (preprocessed/qcied/doubletremoved/featureselected on seurat/scanpy/signac with consensus parameters) of each modality into a latent space with mean and variance. This allows the model to adapt to modality-specific variations in feature distributions. Unlike standard VAEs, where a single Gaussian prior is used, this approach enables better multi-omics integration.


- Fusion via a Shared Latent Space (Latent Space Fusion)

Each omics type passes through a separate encoder but contributes to a shared latent space. The decoder reconstructs multiple outputs from this fused space, enabling joint representation learning. This fusion mechanism ensures that modalities influence each other while retaining modality-specific characteristics. An advatnage of building fusion model by concatenating different latent spaces is that different omics layers might have vastly different structures (e.g., RNA counts vs. chromatin accessibility), making early or input level fusion challenging.

Maybe will expand incorporating cross-modal attention: Use attention-based architectures to weigh the importance of different modalities dynamically.


The dataset utilized in this project originates from the article "Spatial transcriptomic analysis of primary and metastatic pancreatic cancers highlights tumor microenvironmental heterogeneity," published in Nature Genetics in 2024 (Article link: https://doi.org/10.1038/s41588-024-01914-4).
The original data provided by the authors was stored in an RDS format. For the purposes of this project, we converted it into an h5ad file format and made it publicly accessible through the following repository: https://doi.org/10.6084/m9.figshare.28835765.v1.


... to be completed 
