#!/usr/bin/env Rscript

## Purpose: Single-cell or 10x spatial transcriptomics datsets

# ### 10x spatial dataset
# args.root/
#   spatial/
#     scalefactors_json.json
#     tissue_hires_image.png
#     tissue_lowres_image.png
#     tissue_positions_list.csv
#   filtered_count_matrix/
#     barcodes.tsv.gz
#     features.tsv.gz
#     matrix.mtx.gz
#   filtered_feature_bc_matrix.h5

# `tissue_lowres_image.png` is not strictly required.
# At least one of `filtered_feature_bc_matrix.h5` or
# the `filtered_count_matrix` folder should be provided;
# both can exist simultaneously.



# ### Single-cell dataset
# args.root/
#   {sample}_exp.csv
#   {sample}.rds

# The best option is to have the Seurat single-cell rds files that have been preprocessed by the users.
# Otherwise, it is also possible to create it automatically in MMMVAE using {sample}_exp.csv.
# {sample}_exp.csv is a  gene profile matrix  where row is gene and column is single-cell. 





## Usage: source("seurat_to_mmmvae.R")

# species is one of 'Homo_sapiens' and 'Mus_musculus'
# datatype is one of 'single_cell' and 'Spatial'
# sample is sample name of input files, which is also used to name the output file
# root is path of root eg :"D:/Desktop/MMMVAE/test/GSM8599602"
# output_root is path of output_root eg :"D:/Desktop/MMMVAE/test/res"
#'----------------------------------------------------------------------
seurat2mmmvae <- function(sample,species,datatype,root,output_root){
  stopifnot(is.character(sample))
  stopifnot(datatype %in% c("Spatial","single_cell"))
  stopifnot(species %in% c("Homo_sapiens","Mus_musculus"))
  stopifnot(dir.exists(root))

  required_pkgs <- c("Seurat","rio","org.Hs.eg.db","org.Mm.eg.db","clusterProfiler",
                     "irGSEA","dplyr","tidyverse","stringr","Matrix","FNN")
  for(pkg in required_pkgs){
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("Package '%s' is required, please install it first", pkg))
    }
  }

  if (!dir.exists(output_root)) {
    dir.create(output_root, recursive = TRUE)
  }
  if(species=="Homo_sapiens"){
    kegg_gene_list = kegg_gene_list_H
    metabolic_genes <- metabolic_genes_H
    } else if(species=="Mus_musculus"){
    kegg_gene_list <- kegg_gene_list_M
    metabolic_genes <- metabolic_genes_M 
    }
  
  if (datatype =="Spatial"){
  ##############creat seurat object##############
  #Determine the existence of filtered_feature_bc_matrix
  if ("filtered_feature_bc_matrix" %in% 
      list.dirs(root, full.names = FALSE, recursive = FALSE)&&
      length(list.files(file.path(root, "filtered_feature_bc_matrix"))) > 0) {
    counts <- Seurat::Read10X(data.dir = file.path(root,"filtered_feature_bc_matrix"),
                      gene.column = 1)
    if(species=="Homo_sapiens"){
      if(grepl("^ENSG", rownames(counts)[1])){
        ensembl_id <- rownames(counts)
        gene_map <- clusterProfiler::bitr(
          geneID = ensembl_id,
          fromType = "ENSEMBL",
          toType = "SYMBOL",
          OrgDb = org.Hs.eg.db::org.Hs.eg.db, 
          drop = FALSE
        )
        gene_map = na.omit(gene_map[!duplicated(gene_map$SYMBOL),])
        counts = counts[gene_map$ENSEMBL,]
        rownames(counts) = gene_map$SYMBOL
      }
    }else if(species=="Mus_musculus"){
      if(grepl("^ENSMUSG", rownames(counts)[1])){
        ensembl_id <- rownames(counts)
        gene_map <- clusterProfiler::bitr(
          geneID = ensembl_id,
          fromType = "ENSEMBL",
          toType = "SYMBOL",
          OrgDb = org.Mm.eg.db::org.Mm.eg.db,
          drop = FALSE
        )
        gene_map = na.omit(gene_map[!duplicated(gene_map$SYMBOL),])
        counts = counts[gene_map$ENSEMBL,]
        rownames(counts) = gene_map$SYMBOL
      }
    }

    
    st <- Seurat::CreateSeuratObject(counts = counts, assay = "Spatial")
    #Determine the existence of tissue_lowres_image
    if(file.exists(file.path(root, "tissue_lowres_image.png"))){
      image2 <- Seurat::Read10X_Image(
        image.dir = file.path(root, "spatial"),
        image.name = "tissue_lowres_image.png",
        filter.matrix = FALSE  
      )
      DefaultAssay(image2) <- "Spatial"
      image2 <- image2[Cells(x = st)]
      st[["slice1"]] <- image2
    }else {

      img = Seurat::Read10X_Image(
        image.dir = file.path(root, "spatial"),
        image.name = "tissue_hires_image.png", 
        filter.matrix = FALSE
      )
      # st <- CreateSeuratObject(counts = counts, assay = "Spatial")
      DefaultAssay(img) <- "Spatial"
      img <- img[Cells(x = st)]
      st[["slice1"]] <- img
      st@images$slice1@scale.factors$lowres = st@images$slice1@scale.factors$hires
    }
  }else if (file.exists(file.path(root, "filtered_feature_bc_matrix.h5"))) {
    if( file.exists(file.path(root, "spatial/tissue_lowres_image.png"))){
      st <- Seurat::Load10X_Spatial(data.dir = root,
                            filename = "filtered_feature_bc_matrix.h5",
                            image = NULL)
    } else {

      img = Seurat::Read10X_Image(
        image.dir = file.path(root, "spatial"),
        image.name = "tissue_hires_image.png",
        filter.matrix = FALSE 
      )
      st <- Seurat::Load10X_Spatial(data.dir = root,
                            filename = "filtered_feature_bc_matrix.h5",image = img)
      st@images$slice1@scale.factors$lowres = st@images$slice1@scale.factors$hires
    }
  }

  ############## coords##############
  coords <- st@images$slice1$centroids@coords
  rownames(coords) <- st@images$slice1$centroids@cells
  coords <- as.data.frame(coords)
  file = file.path(output_root, paste0(sample, "_coords.csv"))
  rio::export(coords,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file))
  
  
  
  ############## scores##############
  if(nrow(coords)!=ncol(st)){st = st[,rownames(coords)]}

  irGSEAres <- irGSEA::irGSEA.score(object = st, assay = "Spatial",
                            slot = "counts",
                            seeds = 123, 
                            min.cells = 0, min.feature = 0,
                            custom = TRUE, ncores = 1,
                            geneset = kegg_gene_list,
                            msigdb = FALSE,
                            species = species, 
                            geneid = "symbol",
                            method = "AUCell",
                            kcdf = 'Poisson',
                            minGSSize = 1, maxGSSize = 20000)
  score <- tryCatch({
    data.frame(t(irGSEAres@assays$AUCell@scale.data))
  }, error = function(e) {
    data.frame(t(irGSEAres@assays$AUCell@layers$scale.data))
  })
  rownames(score) = colnames(st)
  colnames(score) = names(kegg_gene_list)
  
  

  file = file.path(output_root, paste0(sample, "_score.csv"))
  rio::export(score,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file))


  
  
  ##############Adj_matrix##############
  n <- nrow(coords)
  spot_names <- rownames(coords)
  coords_mat <- as.matrix(coords[, c("x", "y")])
  A <- Matrix::Matrix(0, n, n,dimnames = list(spot_names, spot_names))
  k <- 6
  for (i in 1:n) {
    dx <- coords_mat[,1] - coords_mat[i,1]
    dy <- coords_mat[,2] - coords_mat[i,2]
    dist2 <- dx^2 + dy^2
    idx <- order(dist2)[2:(k+1)]
    sigma <- mean(dist2[idx])
    sigma <- ifelse(sigma == 0, 1, sigma)
    weight <- exp(-dist2[idx] / (2 * sigma))
    A[i, idx] <- weight
  }
  A <- (A + Matrix::t(A)) / 2
  A_hat <- A + Matrix::Diagonal(n)
  deg <- Matrix::rowSums(A_hat)
  deg[deg == 0] <- 1
  D_inv_sqrt <- Matrix::Diagonal(n, 1 / sqrt(deg))
  adj_matrix <- D_inv_sqrt %*% A_hat %*% D_inv_sqrt
  adj_matrix = data.frame(adj_matrix)
  
  colnames(adj_matrix) = colnames(st)
  rownames(adj_matrix) = colnames(st)
  
  file = file.path(output_root, paste0(sample, "_adj_matrix.csv"))
  rio::export(adj_matrix,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file))
  
  
  ##############exp##############
  exp = data.frame(st@assays$Spatial$counts)
  
  file = file.path(output_root, paste0(sample, "_exp.csv"))
  rio::export(exp,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file))
  
  ##############counts##############
  genes = intersect(rownames(exp) , unique(unlist(metabolic_genes)))
  metacounts =data.frame(t(exp[genes,]))
  
  file = file.path(output_root, paste0(sample, "_counts.csv"))
  rio::export(metacounts,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file)) 
  
  ##############neighbor_means##############
  nn  <- FNN::get.knnx(coords, coords, k = 7)   # 7-11=6 
  neighbor_means <- matrix(0, 
                           nrow = nrow(metacounts), 
                           ncol = ncol(metacounts),
                           dimnames = list(rownames(metacounts), colnames(metacounts)))
  for (cell_idx in seq_len(nrow(metacounts))) {
    valid_indices <- nn$nn.index[cell_idx, -1]
    if (length(valid_indices) > 0) {
      neighbor_means[cell_idx, ] <- colMeans(metacounts[valid_indices, , drop = FALSE],
                                             na.rm = TRUE)
    } else {
      neighbor_means[cell_idx, ] <- 0
    }
  }

  file = file.path(output_root, paste0(sample, "_neighbor_means_counts.csv"))
  rio::export(neighbor_means,file,row.names = TRUE)
  message(sprintf("Finished! Output file: %s", file))
  
  
  
  #####################Single-cell######################
  }else if (datatype =="single_cell"){
    rds_file <- file.path(root, paste0(sample, ".rds"))
    csv_file <- file.path(root, paste0(sample, "_exp.csv"))
    
    if (file.exists(rds_file)) {
      sc <- readRDS(rds_file)
      ##############exp
      exp = data.frame(sc@assays$RNA$counts)
      file = file.path(output_root, paste0(sample, "_exp.csv"))
      rio::export(exp,file,row.names = TRUE)
    } else if (file.exists(csv_file)){
      exp <- read.csv(csv_file,row.names =1 ,header = T)
      if(is.numeric(rownames(exp)[1])) stop("The first column should be labeled as Gene Symbol")
      if(is.null(colnames(exp))) stop("Missing sample column name")
      
      
      file = file.path(output_root, paste0(sample, "_exp.csv"))
      rio::export(exp,file,row.names = TRUE)
      message(sprintf("Finished! Output file: %s", file))
      
      sc <- Seurat::CreateSeuratObject(counts = exp)
      sc <- Seurat::NormalizeData(sc, normalization.method="LogNormalize")
      sc <- Seurat::FindVariableFeatures(sc,nfeatures = 2000,)
      sc <- Seurat::ScaleData(sc, features = rownames(sc))
      sc <- Seurat::RunPCA(sc, features = Seurat::VariableFeatures(object = sc))
      sc <- Seurat::FindNeighbors(sc, dims = 1:20)
      sc <- Seurat::FindClusters(sc, resolution = 0.5)
      sc <- Seurat::RunUMAP(sc, dims = 1:20)
    }
    ##############coords
    coords <-Seurat::Embeddings(sc, "umap") 
    colnames(coords) = c("x","y")
    
    file = file.path(output_root, paste0(sample, "_coords.csv"))
    rio::export(coords,file,row.names = TRUE)
    message(sprintf("Finished! Output file: %s", file))
    
    
    
    ############## scores
    irGSEAres <- irGSEA::irGSEA.score(object = sc, assay = "RNA",
                              slot = "counts",
                              seeds = 123, 
                              min.cells = 0, min.feature = 0,
                              custom = TRUE, ncores = 1,
                              geneset = kegg_gene_list,
                              msigdb = FALSE,
                              species = species, 
                              geneid = "symbol",
                              method = "AUCell",
                              kcdf = 'Poisson',
                              minGSSize = 1, maxGSSize = 20000)
    score <- tryCatch({
      data.frame(t(irGSEAres@assays$AUCell@scale.data))
    }, error = function(e) {
      data.frame(t(irGSEAres@assays$AUCell@layers$scale.data))
    })
    rownames(score) = colnames(sc)
    colnames(score) = names(kegg_gene_list)
    
    file = file.path(output_root, paste0(sample, "_score.csv"))
    rio::export(score,file,row.names = TRUE)
    message(sprintf("Finished! Output file: %s", file))
    
    
    
    
    ##############counts
    genes = intersect(rownames(exp) , unique(unlist(metabolic_genes)))
    metacounts =data.frame(t(exp[genes,]))
    
    file = file.path(output_root, paste0(sample, "_counts.csv"))
    rio::export(metacounts,file,row.names = TRUE)
    message(sprintf("Finished! Output file: %s", file)) 
  }
  return(invisible(list(output_root = output_root)))
}


### END！
