# About

The Docker image is built to convert initial spatial and single-cell datasets from RDS format to H5AD format.

## Requirements

The analysis requires several R packages for spatial transcriptomics and single-cell analysis. We provide a Docker container with all dependencies pre-installed.

## Docker Container

This project includes a Docker/Apptainer container with all required dependencies pre-installed.

### Using the container

```bash
# Pull from Docker Hub
docker pull piotto/seurat_env:latest

# With Apptainer/Singularity on HPC
apptainer pull seurat_env.sif docker://piotto/seurat_env:latest
apptainer run --no-home --bind /path/to/data:/data seurat_env.sif
