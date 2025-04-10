
library(zellkonverter)
library(Seurat)
library(SingleCellExperiment)

#given visium_obj is a processed spatial object with seurat 
#function for converting spatial to sce 
convert_spatial_to_sce <- function(visium_obj,
                                   image_name = "anterior1",
                                   output_h5ad = "spatial_data.h5ad",
                                   save_image = TRUE,
                                   image_filename = "spatial_image.png",
                                   image_width = 600,
                                   image_height = 600) {
  
  # Convert to SingleCellExperiment
  sce_obj <- as.SingleCellExperiment(visium_obj)
  
  # Extract spatial coordinates from the Visium object (visium2 here)
  spatial_coords <- visium_obj@images[[image_name]]@boundaries$centroids@coords
  
  # Add to reducedDims
  reducedDims(sce_obj)$spatial <- spatial_coords
  
  # Add image metadata
  metadata(sce_obj)$spatial_image <- list(
    scale_factors = visium_obj@images[[image_name]]@scale.factors
  )
  
  # Save the image to disk if requested
  if (save_image) {
    png(image_filename, width = image_width, height = image_height)
    plot(visium_obj@images[[image_name]]@image)
    dev.off()
  }
  
  # Save the object as H5AD
  writeH5AD(sce_obj, file = output_h5ad)
  
  # Return the SingleCellExperiment object
  return(sce_obj)
}

# Using with your object
sce_brain <- convert_spatial_to_sce(object,
                                    image_name = "anterior1",
                                    output_h5ad = "stxBrain_spatial.h5ad",
                                    image_filename = "stxBrain_image.png")
