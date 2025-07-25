nextflow.enable.dsl = 2

//-------------------------------------------------
// PARAMETERS
//-------------------------------------------------
params.input  = params.input  ?: 'data/spatial_with_images.h5ad'
params.outdir = params.outdir ?: 'results'

//-------------------------------------------------
// PROCESSES
//-------------------------------------------------
process GRAPH {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path h5ad
        
    output:
        path 'slide_graphs.pkl' , emit: graphs
        path 'valid_masks.pkl'  , emit: masks
        
    script:
    """
    python3 ${projectDir}/scripts/construct_graphs.py \
            --input $h5ad \
            --output .
    """
}

process EXTRACT_FEATURES {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path h5ad
        
    output:
        path 'image_features.npy', emit: feats
        
    script:
    """
    python3 ${projectDir}/scripts/extract_image_features.py \
            --input $h5ad \
            --output image_features.npy
    """
}

process TRAIN {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path h5ad
        path adj
        path mask
        path feats
        
    output:
        path 'model.h5', emit: model
        
    script:
    """
    python3 ${projectDir}/scripts/gnn_training.py \
            --input $h5ad \
            --adj   $adj \
            --mask  $mask \
            --image_features $feats \
            --output_model model.h5
    """
}

process LATENT {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path h5ad
        path model_h5
        
    output:
        path 'latent.npy', emit: lat
        
    script:
    """
    python3 ${projectDir}/scripts/latent_extraction.py \
            --input $h5ad \
            --model $model_h5 \
            --output latent.npy
    """
}

process UMAP {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path latent
        
    output:
        path 'umap.png', emit: umap_png
        
    script:
    """
    python3 ${projectDir}/scripts/umap_projection.py \
            --input $latent \
            --output umap.png
    """
}

process ANALYSIS {
    publishDir "${params.outdir}", mode: 'copy'
    conda 'gnn-env'
    
    input:
        path umap_png
        
    output:
        path 'metrics.json'
        
    script:
    """
    python3 ${projectDir}/scripts/training_analysis.py \
            --umap $umap_png \
            --output metrics.json
    """
}

//-------------------------------------------------
// WORKFLOW
//-------------------------------------------------
workflow {
    // Create input channel
    h5ad_ch = Channel.fromPath(params.input)
    
    // Execute processes with proper dependencies
    GRAPH(h5ad_ch)
    EXTRACT_FEATURES(h5ad_ch)
    
    TRAIN(
        h5ad_ch,
        GRAPH.out.graphs,
        GRAPH.out.masks,
        EXTRACT_FEATURES.out.feats
    )
    
    LATENT(
        h5ad_ch,
        TRAIN.out.model
    )
    
    UMAP(LATENT.out.lat)
    
    ANALYSIS(UMAP.out.umap_png)
}
