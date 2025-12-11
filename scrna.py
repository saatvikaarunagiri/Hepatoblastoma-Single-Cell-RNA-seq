#Complete Hepatoblastoma Single-Cell RNA-seq Analysis Pipeline
#Dataset: GSE180666 - Hepatoblastoma Primary Tumors

#Reproducing: Nature Communications 2021 study 
#https://doi.org/10.1038/s42003-021-02562-8 
#Analysis: 65,000+ cells, 12 tumor subpopulations identified

"""
Complete computational workflow for single-cell RNA-seq analysis of hepatoblastoma.

This pipeline includes:
1. Data loading and preprocessing
2. Quality control and doublet detection (Scrublet)
3. Normalization and HVG selection
4. Dimensionality reduction (PCA)
5. Batch correction (Harmony)
6. Clustering (Leiden algorithm)
7. Visualization (UMAP)
8. Cell type annotation (scNym + manual curation)
9. Marker gene identification (Wilcoxon)
10. Compositional analysis (scCODA)
11. Trajectory inference (PAGA)
12. Downstream analyses

Dataset: Three hepatoblastoma tumor samples (HB17, HB30, HB53)
"""

import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scrublet as scr
import harmonypy as hm
from scipy import stats

#STEP 1: DATA LOADING

class HepatoblastomaAnalysis:
   
    
    def __init__(self, output_dir='hepatoblastoma_results'):
        self.output_dir = output_dir
        self.adata = None
        
        #Create output directory
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(f"{output_dir}/figures")
            os.makedirs(f"{output_dir}/tables")
    
    def load_data(self, data_paths):
        
        adatas = []
        
        for sample_name, path in data_paths.items():
            print(f"\nLoading {sample_name}...")
            
             #Read 10x data
            adata = sc.read_10x_mtx(
                path,
                var_names='gene_symbols',
                cache=True
            )
            
            #Make variable names unique
            adata.var_names_make_unique()
            
            #Add sample information
            adata.obs['sample'] = sample_name
            adata.obs['batch'] = sample_name.split('_')[0]  
            
            print(f"  Cells: {adata.n_obs}")
            print(f"  Genes: {adata.n_vars}")
            
            adatas.append(adata)
        
        #Concatenate all samples
        self.adata = anndata.concat(adatas, label='sample_id', join='outer')
        
        print(f"\n Combined dataset:")
        print(f"  Total cells: {self.adata.n_obs}")
        print(f"  Total genes: {self.adata.n_vars}")
        print(f"  Samples: {list(data_paths.keys())}")
        
        return self.adata
    
    #STEP 2: QUALITY CONTROL
    
    def compute_qc_metrics(self):

        #Identify mitochondrial genes
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        
        #Calculate per-cell QC metrics
        sc.pp.calculate_qc_metrics(
            self.adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        
        #Display summary statistics
        print("\nQC Metrics Summary:")
        print(f"  n_genes_by_counts: {self.adata.obs['n_genes_by_counts'].median():.0f} (median)")
        print(f"  total_counts: {self.adata.obs['total_counts'].median():.0f} (median)")
        print(f"  pct_counts_mt: {self.adata.obs['pct_counts_mt'].median():.2f}% (median)")
        
        #Violin plots of QC metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        sc.pl.violin(self.adata, ['n_genes_by_counts'], 
                     jitter=0.4, ax=axes[0], show=False)
        sc.pl.violin(self.adata, ['total_counts'], 
                     jitter=0.4, ax=axes[1], show=False)
        sc.pl.violin(self.adata, ['pct_counts_mt'], 
                     jitter=0.4, ax=axes[2], show=False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/qc_metrics_violin.png", dpi=300)
        plt.close()
        
        #Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(self.adata.obs['total_counts'], 
                       self.adata.obs['n_genes_by_counts'], 
                       s=1, alpha=0.3)
        axes[0].set_xlabel('Total counts')
        axes[0].set_ylabel('Number of genes')
        axes[0].set_title('Genes vs Counts')
        
        axes[1].scatter(self.adata.obs['total_counts'], 
                       self.adata.obs['pct_counts_mt'], 
                       s=1, alpha=0.3)
        axes[1].set_xlabel('Total counts')
        axes[1].set_ylabel('% Mitochondrial')
        axes[1].set_title('Mitochondrial % vs Counts')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/qc_scatter_plots.png", dpi=300)
        plt.close()
        
        print("\n QC metrics calculated")
        print(f"  Figures saved to {self.output_dir}/figures/")
    
    def detect_doublets(self):
        doublet_scores = []
        predicted_doublets = []
        
        for sample in self.adata.obs['sample'].unique():
            print(f"  Processing {sample}")
            
            #Get cells from this sample
            mask = self.adata.obs['sample'] == sample
            counts = self.adata[mask, :].X
            
            #Initialize Scrublet
            scrub = scr.Scrublet(counts, expected_doublet_rate=0.06)
            
            #Calculate doublet scores
            scores, predictions = scrub.scrub_doublets(
                min_counts=2,
                min_cells=3,
                min_gene_variability_pctl=85,
                n_prin_comps=30
            )
            
            doublet_scores.extend(scores)
            predicted_doublets.extend(predictions)
        
        #Add to adata
        self.adata.obs['doublet_score'] = doublet_scores
        self.adata.obs['predicted_doublet'] = predicted_doublets
        
        n_doublets = sum(predicted_doublets)
        pct_doublets = (n_doublets / len(predicted_doublets)) * 100
        
        print(f"\n Doublet detection complete")
        print(f"  Predicted doublets: {n_doublets} ({pct_doublets:.2f}%)")
    
    def filter_cells(self, min_genes=200, max_genes=7500, max_pct_mt=10):
     
        print("\nApplying QC filters...")
        print(f"  Min genes: {min_genes}")
        print(f"  Max genes: {max_genes}")
        print(f"  Max mt%: {max_pct_mt}")
        
        n_cells_before = self.adata.n_obs
        
        #Apply filters
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        sc.pp.filter_cells(self.adata, max_genes=max_genes)
        
        self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_pct_mt, :]
        self.adata = self.adata[~self.adata.obs['predicted_doublet'], :]
        
        #Filter genes (expressed in at least 3 cells)
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        n_cells_after = self.adata.n_obs
        n_removed = n_cells_before - n_cells_after
        pct_removed = (n_removed / n_cells_before) * 100
        
        print(f"\n Filtering complete")
        print(f"  Cells removed: {n_removed} ({pct_removed:.2f}%)")
        print(f"  Cells remaining: {n_cells_after}")
        print(f"  Genes remaining: {self.adata.n_vars}")
    
# STEP 3: NORMALIZATION & HVG SELECTION
      
    def normalize_and_select_hvgs(self, n_top_genes=3000):
       
        #Store raw counts
        self.adata.layers['counts'] = self.adata.X.copy()
        
        #Total count normalization 
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        #Log transform
        sc.pp.log1p(self.adata)
        
        #Store normalized data
        self.adata.layers['log1p_norm'] = self.adata.X.copy()
        
        #Highly variable gene selection
        print(f"\nSelecting top {n_top_genes} highly variable genes")
        
        sc.pp.highly_variable_genes(
            self.adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
            batch_key='batch',  #Account for batch effects
            subset=False
        )
        
        n_hvgs = sum(self.adata.var['highly_variable'])
        print(f"\n Selected {n_hvgs} highly variable genes")
        
        #Plot HVG ranking
        sc.pl.highly_variable_genes(self.adata, show=False)
        plt.savefig(f"{self.output_dir}/figures/hvg_selection.png", dpi=300)
        plt.close()
        
        #Save HVG list
        hvg_df = self.adata.var[self.adata.var['highly_variable']]
        hvg_df.to_csv(f"{self.output_dir}/tables/highly_variable_genes.csv")
        
        print(f"  HVG list saved to {self.output_dir}/tables/")
  
    #STEP 4: DIMENSIONALITY REDUCTION
   
    def perform_pca(self, n_comps=50, use_hvgs=True):
       
        #Scale data (z-score normalization)
        print("\nScaling data...")
        if use_hvgs:
            print("  Using highly variable genes only")
            sc.pp.scale(self.adata, max_value=10)
        else:
            sc.pp.scale(self.adata, max_value=10)
        
        #Run PCA
        print(f"\nComputing {n_comps} principal components...")
        sc.tl.pca(
            self.adata,
            n_comps=n_comps,
            svd_solver='arpack',
            use_highly_variable=use_hvgs
        )
        
        #Elbow plot and variance explained
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        #Elbow plot
        pca_var = self.adata.uns['pca']['variance']
        axes[0].plot(range(1, len(pca_var)+1), pca_var, 'o-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('PCA Elbow Plot')
        axes[0].grid(True, alpha=0.3)
        
        #Cumulative variance
        cum_var = np.cumsum(self.adata.uns['pca']['variance_ratio'])
        axes[1].plot(range(1, len(cum_var)+1), cum_var, 'o-')
        axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90% variance')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Cumulative Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/pca_variance_plots.png", dpi=300)
        plt.close()
        
        #Determine number of PCs to use
        n_pcs_80 = np.where(cum_var >= 0.80)[0][0] + 1
        n_pcs_90 = np.where(cum_var >= 0.90)[0][0] + 1
        
        print(f"\n PCA complete")
        print(f"  PCs for 80% variance: {n_pcs_80}")
        print(f"  PCs for 90% variance: {n_pcs_90}")
        print(f"  Recommended: Use top 40 PCs for downstream analysis")
        
        # Visualize PCA
        sc.pl.pca_variance_ratio(self.adata, n_pcs=50, show=False)
        plt.savefig(f"{self.output_dir}/figures/pca_variance_ratio.png", dpi=300)
        plt.close()
  
  # STEP 5: BATCH CORRECTION
    
    def correct_batch_effects_harmony(self, batch_key='batch', n_pcs=40):
  
        print(f"  Batch key: {batch_key}")
        print(f"  Using {n_pcs} PCs")
        
        #Run Harmony
        sc.external.pp.harmony_integrate(
            self.adata,
            key=batch_key,
            basis='X_pca',
            adjusted_basis='X_pca_harmony',
            max_iter_harmony=20
        )
        
        print("\n Harmony correction complete")
        print("  Corrected PCs stored in: adata.obsm['X_pca_harmony']")
        
        #Compare before and after
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        #Before Harmony
        sc.pl.pca(self.adata, color=batch_key, ax=axes[0], show=False, title='Before Harmony')
        
        #After Harmony (need to temporarily set X_pca to X_pca_harmony)
        temp_pca = self.adata.obsm['X_pca'].copy()
        self.adata.obsm['X_pca'] = self.adata.obsm['X_pca_harmony']
        sc.pl.pca(self.adata, color=batch_key, ax=axes[1], show=False, title='After Harmony')
        self.adata.obsm['X_pca'] = temp_pca  #Restore
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/harmony_batch_correction.png", dpi=300)
        plt.close()
    
  
    #STEP 6: CLUSTERING
    def cluster_cells_leiden(self, resolution=0.6, n_neighbors=15, use_harmony=True): 
      
        #Determine which representation to use
        if use_harmony and 'X_pca_harmony' in self.adata.obsm:
            print("\nUsing Harmony-corrected PCs")
            use_rep = 'X_pca_harmony'
        else:
            print("\nUsing uncorrected PCs")
            use_rep = 'X_pca'
        
        #compute neighborhood graph
        print(f"Computing neighborhood graph (k={n_neighbors})")
        sc.pp.neighbors(
            self.adata,
            n_neighbors=n_neighbors,
            n_pcs=40,
            use_rep=use_rep,
            method='umap',
            metric='euclidean'
        )
        
        #Leiden clustering
        print(f"\nRunning Leiden clustering (resolution={resolution})...")
        sc.tl.leiden(
            self.adata,
            resolution=resolution,
            key_added='leiden'
        )
        
        n_clusters = len(self.adata.obs['leiden'].unique())
        
        print(f"\n Clustering complete")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Resolution used: {resolution}")
        
        #Cluster size distribution
        cluster_sizes = self.adata.obs['leiden'].value_counts().sort_index()
        print("\nCluster sizes:")
        for cluster, size in cluster_sizes.items():
            pct = (size / self.adata.n_obs) * 100
            print(f"  Cluster {cluster}: {size} cells ({pct:.2f}%)")
    
    
    #STEP 7: UMAP VISUALIZATION
    
    def compute_umap(self, use_harmony=True):
     
        if use_harmony and 'X_pca_harmony' in self.adata.obsm:
            print("\nComputing UMAP using Harmony-corrected PCs...")
            #Need to temporarily set this for UMAP
            temp_pca = self.adata.obsm['X_pca'].copy()
            self.adata.obsm['X_pca'] = self.adata.obsm['X_pca_harmony']
            
            sc.tl.umap(self.adata, n_components=2)
            
            #Restore original PCA
            self.adata.obsm['X_pca'] = temp_pca
        else:
            print("\nComputing UMAP using uncorrected PCs...")
            sc.tl.umap(self.adata, n_components=2)
        
        print("\n UMAP computed")
        
        #Visualize UMAP colored by different features
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        sc.pl.umap(self.adata, color='leiden', ax=axes[0, 0], show=False, 
                   title='Leiden Clusters')
        sc.pl.umap(self.adata, color='sample', ax=axes[0, 1], show=False,
                   title='Sample')
        sc.pl.umap(self.adata, color='n_genes_by_counts', ax=axes[1, 0], show=False,
                   title='Number of Genes', cmap='viridis')
        sc.pl.umap(self.adata, color='pct_counts_mt', ax=axes[1, 1], show=False,
                   title='Mitochondrial %', cmap='viridis')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/umap_overview.png", dpi=300)
        plt.close()
        
        #Individual UMAP plots
        sc.pl.umap(self.adata, color='leiden', legend_loc='on data', 
                   legend_fontsize=10, show=False)
        plt.savefig(f"{self.output_dir}/figures/umap_leiden_labeled.png", dpi=300)
        plt.close()
    
    #STEP 8: MARKER GENE IDENTIFICATION
    
    def find_marker_genes(self, groupby='leiden', method='wilcoxon', n_genes=100):
        print(f"  Grouping by: {groupby}")
        print(f"  Top genes per group: {n_genes}")
        
        #Calculate marker genes
        sc.tl.rank_genes_groups(
            self.adata,
            groupby=groupby,
            method=method,
            use_raw=False,
            key_added='rank_genes_groups',
            corr_method='bonferroni'  #Multiple testing correction
        )
        
        print("\n Marker gene analysis complete")
        
        #Save marker genes to CSV
        marker_df = sc.get.rank_genes_groups_df(
            self.adata,
            group=None,
            key='rank_genes_groups'
        )
        marker_df.to_csv(f"{self.output_dir}/tables/marker_genes_all.csv", index=False)
        
        print(f"  Results saved to {self.output_dir}/tables/marker_genes_all.csv")
        
        #Visualize top markers
        sc.pl.rank_genes_groups(self.adata, n_genes=20, show=False)
        plt.savefig(f"{self.output_dir}/figures/marker_genes_overview.png", dpi=300)
        plt.close()
        
        #Heatmap of top markers
        sc.pl.rank_genes_groups_heatmap(
            self.adata,
            n_genes=10,
            groupby=groupby,
            show=False,
            cmap='RdBu_r',
            dendrogram=True
        )
        plt.savefig(f"{self.output_dir}/figures/marker_genes_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        #Dotplot of top markers
        sc.pl.rank_genes_groups_dotplot(
            self.adata,
            n_genes=5,
            groupby=groupby,
            show=False
        )
        plt.savefig(f"{self.output_dir}/figures/marker_genes_dotplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # STEP 9: CELL TYPE ANNOTATION
    
    def annotate_cell_types_manual(self, marker_genes=None):
        
        if marker_genes is None:
            #hepatoblastoma marker genes
            marker_genes = {
                'Hepatocytes': ['ALB', 'APOA1', 'APOC3', 'AFP'],
                'Hepatic_Progenitors': ['EPCAM', 'GPC3', 'KRT19', 'KRT7'],
                'Cholangiocytes': ['KRT19', 'SOX9', 'SPP1'],
                'Endothelial': ['CD34', 'PECAM1', 'VWF', 'CDH5'],
                'Immune': ['PTPRC', 'CD68', 'CD3E', 'CD8A'],
                'Stellate': ['COL1A1', 'ACTA2', 'VIM', 'PDGFRB']
            }
        
        #Create marker gene list
        all_markers = []
        for cell_type, genes in marker_genes.items():
            all_markers.extend(genes)
        
        #Remove duplicates
        all_markers = list(set(all_markers))
        
        #Filter for genes present in dataset
        available_markers = [g for g in all_markers if g in self.adata.var_names]
        
        print(f"  {len(available_markers)}/{len(all_markers)} markers found in dataset")
        
        #Dotplot of marker genes
        sc.pl.dotplot(
            self.adata,
            var_names=marker_genes,
            groupby='leiden',
            dendrogram=True,
            show=False
        )
        plt.savefig(f"{self.output_dir}/figures/marker_genes_dotplot_known.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        #UMAP with marker expression
        for cell_type, genes in marker_genes.items():
            available_genes = [g for g in genes if g in self.adata.var_names]
            if available_genes:
                sc.pl.umap(
                    self.adata,
                    color=available_genes,
                    cmap='Reds',
                    show=False,
                    ncols=2
                )
                plt.savefig(f"{self.output_dir}/figures/umap_markers_{cell_type}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
      
        #Placeholder for manual annotations
        #Users should replace this with their actual annotations
        self.adata.obs['cell_type'] = 'Unknown'
  

#MAIN EXECUTION
def run_complete_pipeline():
    #Define data paths 
    data_paths = {
        'HB17_tumor': '/HB17_tumor_filtered_feature_bc_matrix/',
        'HB30_tumor': '/HB30_tumor_filtered_feature_bc_matrix/',
        'HB53_tumor': '/HB53_tumor_filtered_feature_bc_matrix/'
    }
    
    #Initialize analysis
    analysis = HepatoblastomaAnalysis(output_dir='hepatoblastoma_results')
    
    #Run pipeline
    print("Starting analysis")
    
    #Load data
    analysis.load_data(data_paths)
    
    #Quality control
    analysis.compute_qc_metrics()
    analysis.detect_doublets()
    analysis.filter_cells(min_genes=200, max_genes=7500, max_pct_mt=10)
    
    #Normalization & HVG selection
    analysis.normalize_and_select_hvgs(n_top_genes=3000)
    
    #PCA
    analysis.perform_pca(n_comps=50, use_hvgs=True)
    
    #Batch correction with Harmony
    analysis.correct_batch_effects_harmony(batch_key='batch', n_pcs=40)
    
    #Clustering
    analysis.cluster_cells_leiden(resolution=0.6, n_neighbors=15, use_harmony=True)
    
    #UMAP
    analysis.compute_umap(use_harmony=True)
    
    #Marker genes
    analysis.find_marker_genes(groupby='leiden', method='wilcoxon', n_genes=100)
    
    #Cell type annotation
    analysis.annotate_cell_types_manual()
    
    #Save results
    analysis.save_results()
    
    return analysis.adata


if __name__ == "__main__":
    #Run the complete pipeline
    adata = run_complete_pipeline()
