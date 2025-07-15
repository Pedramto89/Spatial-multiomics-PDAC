## 🚀 What Happens in This Notebook  
1. **Checkpoint Load** – pull processed `AnnData`, image features, spatial graphs, and training config generated in NB 1.  
2. **Model Re‑Creation** – rebuild the GNN (2 × `SimpleGraphConvLayer` → 64‑unit head) with dual outputs:  
   * **`class_out`** – binary control (“Is this slide IU_PDA_HM9?”)  
   * **`type_out`** – biologically relevant 4‑class stage (NP, T, HM, LNM)  
3. **Leak‑Proof Data Split** – 24 slides (73 k spots) for training, 6 slides (18 k spots) for validation; no inter‑slide mixing.  
4. **Class‑Imbalance Handling** – compute slide‑aware class weights (≈ 98 % vs 2 % imbalance for `class_out`).  
5. **Training (20 epochs, Adam 1e‑5)**  
   * Batch 32, 2 300 steps/epoch  
   * Early checkpointing to `best_gnn_model_fixed/`  
6. **Metrics Tracking** – log total loss + per‑head loss/accuracy each epoch and plot:  
   * **`class_out_accuracy`** quickly saturates at ≥ 0.97 (diagnostic only)  
   * **`type_out_accuracy`** climbs steadily from ≈ 0.39 → 0.45 (train) and ≈ 0.35 → 0.40 (val).  
7. **Post‑Training Evaluation**  
   * Generate confusion matrix on 2 000 validation spots (sampled)  
   * Compute per‑class precision/recall/F1  
   * Project last hidden layer with UMAP for biological interpretation.  
8. **Save Artifacts** – best weights (`model.h5`), adjacency matrices, and 64‑D embeddings for downstream analysis.

---

## 📊 Key Results  

| Metric (Validation) | NP | T | HM | LNM | Macro Avg |
|---------------------|----|---|----|-----|-----------|
| **Accuracy**        | —  | — | —  | —  | **0.4029** |
| **Recall**          | 0.43 | 0.33 | 0.38 | **0.50** | 0.41 |
| **Precision**       | 0.45 | 0.39 | 0.42 | **0.56** | 0.46 |
| **F1‑score**        | 0.44 | 0.36 | 0.40 | **0.53** | 0.43 |

* **Best‑classified stage = LNM** – lymph‑node metastases exhibit distinct spatial‑transcriptomic signatures.  
* **Main confusion = T ↔ HM (≈ 42 % of T errors)** – hepatic metastases retain many primary‑tumour features.  
* **Overall type accuracy ≈ 40 %** – markedly above random (25 %) on a 4‑class task.  
* **UMAP embedding** separates NP from metastatic phenotypes, evidencing learned biological structure.

---

## 🧬 Interpretation & Take‑aways  
* **Spatial context + multimodal features** genuinely help: gene‑only or image‑only baselines plateaued at ≈ 28–30 % accuracy.  
* **Graph‑convolution layers** capture micro‑environment cues that distinguish metastatic niches from primary tumour cores.  
* **Model limitations** stem from severe class imbalance and morphological similarity between T and HM; attention mechanisms or edge‑weighting by ligand–receptor scores may improve discrimination.  

---

### ➡️ Next Steps  
* Incorporate **Graph Attention Networks (GAT)** to weigh neighbour influence adaptively.  
* Add **edge features** (distance, ligand‑receptor scores) and experiment with **edge‑weighted adjacency**.  
* Fine‑tune on slide‑held‑out test set to verify generalisation.  
* Explore **GVAE / Diffusion‑based** generative models for synthetic spatial‑omics data.
