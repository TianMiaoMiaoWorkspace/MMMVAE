# Code Rscript
seurat_to_mmmvae.R
## Input file hierarchy

### 10x spatial dataset
```markdown
args.root/
    spatial/
        scalefactors_json.json
        tissue_hires_image.png
        tissue_lowres_image.png
        tissue_positions_list.csv
    filtered_count_matrix/
        barcodes.tsv.gz
        features.tsv.gz
        matrix.mtx.gz
    filtered_feature_bc_matrix.h5

`tissue_lowres_image.png` is not strictly required.
At least one of `filtered_feature_bc_matrix.h5` or the `filtered_count_matrix` folder should be provided; both can exist simultaneously.
```  


### Single-cell dataset
```markdown
args.root/
    {sample}_exp.csv 
    {sample}.rds

The best option is to have the Seurat single-cell rds files that have been preprocessed by the users. Otherwise, it is also possible to create it automatically in MMMVAE using {sample}_exp.csv.
{sample}_exp.csv is a  gene profile matrix  where row is gene and column is single-cell. 

``` 


## Output file hierarchy:
```markdown
args.output_root/
    output_root/
        {sample}_coords.csv
        {sample}_scores.csv
        {sample}_adj_matrix.csv
        {sample}_counts.csv
        {sample}_neighbor_means_counts.csv
        {sample}_exp.csv 
        {sample}_flux.csv
        {sample}_balance.csv

Single-cell will not output the "neighbor_means_counts.csv" file
``` 


## Test:
```markdown
source("./seurat_to_mmmvae.R")
load("./metabolic_data.RData")


seurat2mmmvae(
  sample = "spatial_exampleDATA",
  species = "Mus_musculus",
  datatype = "Spatial",
  root = "./example/spatial_exampleDATA/",
  output_root = "./example/spatial_exampleDATA/res"
)

``` 
