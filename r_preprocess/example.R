#!/usr/bin/env Rscript

source("./seurat_to_mmmvae.R")
load("./metabolic_data.RData")


seurat2mmmvae(
  sample = "spatial_exampleDATA",
  species = "Mus_musculus",
  datatype = "Spatial",
  root = "./example/spatial_exampleDATA/",
  output_root = "./example/spatial_exampleDATA/mmmvaeinput"
)

seurat2mmmvae(
  sample = "single-cell_exampleDATA",
  species = "Homo_sapiens",
  datatype = "single_cell",
  root = "./example/single-cell_exampleDATA/",
  output_root = "./example/single-cell_exampleDATA/mmmvaeinput"
)
