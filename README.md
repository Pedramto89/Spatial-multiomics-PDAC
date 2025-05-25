**Multi-Modal VAE with Modality-Specific Priors**

# Spatial Multi‑Omics Integration in Pancreatic Cancer

**Goal:** Integrate spatial transcriptomics with single‑cell RNA‑seq and ATAC‑seq from pancreatic ductal adenocarcinoma (PDAC) to reveal spatially‑resolved tumor–stroma interactions using a Graph Neural Network–enhanced Multi‑Modal Variational Autoencoder (VAE).

---

## Key Ideas
- **Graph Neural Networks (GNNs)** preserve spatial proximity between cells/spots.
- **Modality‑specific priors** in the VAE respect unique data distributions (spatial, RNA, ATAC).
- **Joint latent space** fuses modalities while retaining biological signals.
- **Explainable AI (e.g., SHAP)** interprets latent features.

---

## Data
- **Source:** Public dataset from *Ateeq M. Khaliq et al.,* **Nat Genet 2024** (30 matched primary & metastatic PDAC samples).
- **Weblink:** https://www.nature.com/articles/s41588-024-01914-4  
- **Status:** Already pre‑processed, quality‑controlled, doublet‑filtered (Seurat/Scanpy/Signac).

---

## Workflow

1. **Download & QC**  
   - Pull spatial, scRNA‑seq, scATAC‑seq matrices.  
   - Confirm QC metrics; filter low‑quality cells/spots.

2. **Graph Construction**  
   - Nodes: cells / spots.  
   - Edges: spatial distance + expression similarity.

3. **Encoder Setup**  
   - Spatial encoder → GNN layers.  
   - RNA & ATAC encoders → dense layers.  
   - Learn **modality‑specific priors**.

4. **Latent Fusion**  
   - Project each modality into a shared latent space.  
   - Minimize reconstruction + KL losses.

5. **Decoder & Reconstruction**  
   - Reconstruct original modalities from latent vector.  
   - Evaluate with held‑out data.

6. **Interpretation**  
   - Apply SHAP to latent features.  
   - Map important dimensions back to tissue coordinates.

7. **Biological Analysis**  
   - Identify spatial gene modules.  
   - Detect tumor microenvironment niches.  
   - Nominate biomarkers & therapeutic targets.

---

## Expected Outcomes
- **Method:** Open‑source framework for spatial multi‑omics integration.  
- **Insights:** Spatially coherent signatures of tumor, stroma, and immune niches.  
- **Applications:** Diagnostics, prognostics, and therapy guidance in PDAC and beyond.
