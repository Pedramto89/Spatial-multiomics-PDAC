FROM rocker/r-ver:4.4.2

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Create both internal and data-mounted library locations
RUN mkdir -p /opt/R_libs
RUN mkdir -p /data/R_libs
ENV R_LIBS_USER=/opt/R_libs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libglpk-dev \
    libglpk40 \
    pkg-config \
    zlib1g-dev \
    libpng-dev \
    patch \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /data

# Configure R to use both the internal and data-mounted library paths
RUN echo ".libPaths(c('/opt/R_libs', '/data/R_libs', .libPaths()))" >> /usr/local/lib/R/etc/Rprofile.site

# Install devtools and BiocManager first - using the standard library path to avoid issues
RUN R -e "install.packages(c('devtools', 'BiocManager', 'remotes'), repos='https://cran.rstudio.com')"

# Install required R packages to the custom library
RUN R -e "install.packages('Seurat', repos='https://cran.rstudio.com', lib='/opt/R_libs')"
RUN R -e "install.packages('SeuratDisk', repos='https://cran.rstudio.com', lib='/opt/R_libs')"
RUN R -e "install.packages('reticulate', repos='https://cran.rstudio.com', lib='/opt/R_libs')"

# Install Bioconductor packages
RUN R -e "BiocManager::install(c('SingleCellExperiment', 'LoomExperiment'), lib='/opt/R_libs')"

# Install Miniconda via reticulate for Python
ENV RETICULATE_MINICONDA_PATH=/opt/reticulate_miniconda
RUN R -e "reticulate::install_miniconda(path='/opt/reticulate_miniconda')"
ENV PATH="/opt/reticulate_miniconda/bin:${PATH}"
RUN R -e "reticulate::use_python('/opt/reticulate_miniconda/envs/r-reticulate/bin/python', required=TRUE)"
RUN R -e "reticulate::py_install('anndata', pip=TRUE)"

# Install sceasy from GitHub using remotes instead of devtools
RUN R -e "remotes::install_github('cellgeni/sceasy')"

# Create a startup script to ensure library paths are set correctly
RUN echo '#!/bin/bash\nR "$@"' > /usr/local/bin/start_r && \
    chmod +x /usr/local/bin/start_r

# Default command to start an R session with the correct library paths
CMD ["/usr/local/bin/start_r"]
