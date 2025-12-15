# Hepatoblastoma Single-Cell RNA-seq Analysis

Single-cell transcriptomics analysis of pediatric liver cancer, identifying tumor heterogeneity and developmental trajectories.

## Getting Started

These instructions will help you run the single-cell analysis pipeline.

### Prerequisites

Before running this project, you need:

* Python 3.8 or higher
* pip package manager
* Jupyter notebook (optional, for interactive analysis)

## Usage

### Main Analysis Pipeline

```
$ python scrna.py
```

This script performs:
* Quality control and preprocessing
* Dimensionality reduction (PCA, UMAP)
* Clustering and cell type annotation
* Differential expression analysis

### Advanced Analyses

```
$ python advance_analysis.py
```

This script performs:
* Trajectory inference (Monocle 3)
* RNA velocity analysis (scVelo)
* Cell-cell communication (CellPhoneDB)
* Pathway enrichment

### Expected Runtime

* Main pipeline: 15-30 minutes
* Advanced analyses: 30-60 minutes

## Results

### Cell Type Composition

* Total cells analyzed: 65,353
* Cell types identified: 10 distinct populations
* Hepatocyte subtypes: 3 (fetal, mature, tumor)
* Immune infiltration: 8.8%

### Key Findings

* Tumor cells show fetal-like transcriptional programs
* Limited immune cell presence suggests immunosuppressive microenvironment
* Clear developmental trajectory from fetal to mature hepatocytes

## Output Files

```
outputs/
├── preprocessed_data.h5ad
├── umap_celltype_annotation.png
├── differential_expression_results.csv
├── trajectory_analysis.png
└── pathway_enrichment_results.csv
```

## Technical Details

* Technology: 10x Genomics single-cell RNA-seq
* Quality filters: 200-6000 genes per cell, <15% mitochondrial content
* Clustering: Leiden algorithm (resolution 0.5)
* Normalization: 10,000 counts per cell, log-transformed

## Additional Information

* Sample type: Pediatric hepatoblastoma tumor tissue
* Analysis framework: Scanpy, Monocle 3, scVelo, CellPhoneDB
* Institution: Boston University
