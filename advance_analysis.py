#Advanced Analyses for Hepatoblastoma scRNA-seq
#Includes: scCODA, PAGA (Trajectory inference), RNA velocity preparation and additional visualizations


import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#scCODA for compositional analysis
try:
    from sccoda.util import comp_ana as ca
    from sccoda.util import cell_composition_data as ccd
    from sccoda.util import data_visualization as viz
    from sccoda.model import dirichlet_models as dm
    SCCODA_AVAILABLE = True
except ImportError:
    print("Warning: scCODA not installed. Install with: pip install sccoda")
    SCCODA_AVAILABLE = False

# COMPOSITIONAL ANALYSIS WITH scCODA
class ComposionalAnalysis:
  
    def __init__(self, adata, output_dir='hepatoblastoma_results'):
        self.adata = adata
        self.output_dir = output_dir
        
    def prepare_data(self, cell_type_col='cell_type', condition_col='sample'):
  
        if not SCCODA_AVAILABLE:
            print("\nERROR: scCODA not installed!")
            print("Install with: pip install sccoda")
            return None
        
        print(f"  Cell type column: {cell_type_col}")
        print(f"  Condition column: {condition_col}")
        
        #Create count table
        cell_counts = pd.crosstab(
            self.adata.obs[condition_col],
            self.adata.obs[cell_type_col]
        )
        
        print(f"\nCell type composition:")
        print(cell_counts)
        
        #Calculate proportions
        cell_props = cell_counts.div(cell_counts.sum(axis=1), axis=0)
        
        #Save composition table
        cell_counts.to_csv(f"{self.output_dir}/tables/cell_type_counts.csv")
        cell_props.to_csv(f"{self.output_dir}/tables/cell_type_proportions.csv")
        
        #Visualize composition
        self.plot_composition(cell_props, condition_col)
        
        return cell_counts
    
    def plot_composition(self, cell_props, condition_col):
        
        #Prepare data for plotting
        plot_data = cell_props.reset_index().melt(
            id_vars=condition_col,
            var_name='cell_type',
            value_name='proportion'
        )
        
        #Stacked barplot
        plt.figure(figsize=(12, 6))
        
        #Create color palette
        n_types = len(cell_props.columns)
        colors = sns.color_palette("husl", n_types)
        
        #Plot
        bottom = np.zeros(len(cell_props))
        for i, cell_type in enumerate(cell_props.columns):
            plt.bar(
                cell_props.index,
                cell_props[cell_type],
                bottom=bottom,
                label=cell_type,
                color=colors[i]
            )
            bottom += cell_props[cell_type]
        
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Cell Type Proportion', fontsize=12)
        plt.title('Cell Type Composition by Sample', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/composition_barplot.png", dpi=300)
        plt.close()
        
        print(f"\n Composition plot saved")
    
    def run_scoda_model(self, cell_counts, reference_cell_type='Hepatocytes'):
        if not SCCODA_AVAILABLE:
            print("\nSkipping scCODA analysis (not installed)")
            return None
        
        print(f"\nRunning scCODA model...")
        print(f"  Reference cell type: {reference_cell_type}")
        
        #Convert to scCODA format
        data = ccd.from_pandas(cell_counts, covariate_columns=[])
        
        #Create model
        model = dm.CompositionalModel(
            data,
            formula="C(condition)", 
            reference_cell_type=reference_cell_type
        )
        
        #Fit using Hamiltonian Monte Carlo
        model.sample_hmc(num_results=20000, num_burnin=5000)
        
        #Get results
        results = model.summary()
        print("\nscCODA Results:")
        print(results)
        
        #Save results
        results.to_csv(f"{self.output_dir}/tables/sccoda_results.csv")
        
        #Plot effects
        viz.boxplots(
            data,
            model=model,
            figsize=(10, 8)
        )
        plt.savefig(f"{self.output_dir}/figures/sccoda_effects.png", dpi=300)
        plt.close()
        
        print("\n scCODA analysis complete")
        
        return results

#TRAJECTORY INFERENCE WITH PAGA
class TrajectoryAnalysis:
    
    def __init__(self, adata, output_dir='hepatoblastoma_results'):
       
        self.adata = adata
        self.output_dir = output_dir
    
    def run_paga(self, groupby='leiden', threshold=0.03):
        print(f"  Grouping by: {groupby}")
        print(f"  Threshold: {threshold}")
        
        #Ensure required data exists
        if 'neighbors' not in self.adata.uns:
            print("  Computing neighbors...")
            sc.pp.neighbors(self.adata, n_pcs=40)
        
        #compute PAGA
        sc.tl.paga(self.adata, groups=groupby)
        
        #Plot PAGA graph
        sc.pl.paga(
            self.adata,
            threshold=threshold,
            show=False,
            title='PAGA Graph',
            node_size_scale=1.5,
            edge_width_scale=0.5
        )
        plt.savefig(f"{self.output_dir}/figures/paga_graph.png", dpi=300)
        plt.close()
        
        print("\n PAGA computed")
        
        #PAGA-initialized UMAP
        print("\nRefining UMAP using PAGA...")
        sc.tl.umap(self.adata, init_pos='paga')
        
        sc.pl.umap(
            self.adata,
            color=groupby,
            legend_loc='on data',
            show=False,
            title='UMAP (PAGA-initialized)'
        )
        plt.savefig(f"{self.output_dir}/figures/umap_paga_init.png", dpi=300)
        plt.close()
        
        #PAGA with UMAP overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sc.pl.paga(self.adata, threshold=threshold, ax=ax1, show=False)
        ax1.set_title('PAGA Graph')
        
        sc.pl.umap(self.adata, color=groupby, ax=ax2, show=False)
        ax2.set_title('UMAP with Clusters')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/paga_umap_combined.png", dpi=300)
        plt.close()
        
        #PAGA path analysis 
        if 'cell_type' in self.adata.obs.columns:
            self.analyze_differentiation_paths()
    
    def analyze_differentiation_paths(self):

        #Get PAGA graph
        paga_groups = self.adata.uns['paga']['groups']
        connectivity = self.adata.uns['paga']['connectivities']
        
        #Find highly connected groups
        print("\nHighly connected cell groups:")
        for i, group1 in enumerate(paga_groups):
            for j, group2 in enumerate(paga_groups[i+1:], start=i+1):
                conn = connectivity[i, j]
                if conn > 0.1:  #Threshold for "high" connectivity
                    print(f"  {group1} <-> {group2}: {conn:.3f}")
        
        print("\n Path analysis complete")
    
    def plot_pseudotime(self, root_cell_type='Hepatic_Progenitors'):
 
        if 'cell_type' not in self.adata.obs.columns:
            print("\nWarning: Cell types not annotated, skipping pseudotime")
            return
        
        print(f"\nComputing pseudotime from {root_cell_type}")
        
        #Find root cell
        root_cells = self.adata.obs['cell_type'] == root_cell_type
        if root_cells.sum() > 0:
            root_idx = np.where(root_cells)[0][0]
            
            #Compute diffusion pseudotime
            sc.tl.dpt(self.adata, n_dcs=15)
            
            #Plot pseudotime
            sc.pl.umap(
                self.adata,
                color='dpt_pseudotime',
                cmap='viridis',
                show=False,
                title=f'Pseudotime (root: {root_cell_type})'
            )
            plt.savefig(f"{self.output_dir}/figures/pseudotime_umap.png", dpi=300)
            plt.close()
            
            print(" Pseudotime computed and plotted")
        else:
            print(f"No cells found for {root_cell_type}")

# ADDITIONAL VISUALIZATIONS
class AdditionalViz:
    
    def __init__(self, adata, output_dir='hepatoblastoma_results'):
       
        self.adata = adata
        self.output_dir = output_dir
    
    def plot_marker_heatmap(self, marker_genes, groupby='cell_type'):
    
        print("\nCreating marker gene heatmap...")
        
        #Flatten marker gene list if dict
        if isinstance(marker_genes, dict):
            all_markers = []
            for genes in marker_genes.values():
                all_markers.extend(genes)
            marker_genes = list(set(all_markers))
        
        #Filter for available genes
        available = [g for g in marker_genes if g in self.adata.var_names]
        
        sc.pl.heatmap(
            self.adata,
            var_names=available,
            groupby=groupby,
            swap_axes=True,
            show=False,
            cmap='RdBu_r',
            dendrogram=True,
            figsize=(10, 12)
        )
        plt.savefig(f"{self.output_dir}/figures/marker_heatmap_comprehensive.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Heatmap created")
    
    def plot_cluster_proportions(self, group_by='sample', color_by='leiden'):
      
        #Calculate proportions
        props = pd.crosstab(
            self.adata.obs[group_by],
            self.adata.obs[color_by],
            normalize='index'
        )
        
        #Create stacked barplot
        props.plot(kind='bar', stacked=True, figsize=(12, 6),
                  colormap='tab20')
        plt.xlabel(group_by.capitalize(), fontsize=12)
        plt.ylabel('Proportion', fontsize=12)
        plt.title(f'{color_by.capitalize()} Proportions by {group_by.capitalize()}',
                 fontsize=14, fontweight='bold')
        plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/cluster_proportions_by_{group_by}.png",
                   dpi=300)
        plt.close()
        
        print(" Proportion plot created")
    
    def create_qc_summary_plot(self):  
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        #Cells per sample
        ax1 = fig.add_subplot(gs[0, 0])
        sample_counts = self.adata.obs['sample'].value_counts()
        sample_counts.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Cells per Sample')
        ax1.set_ylabel('Number of Cells')
        ax1.tick_params(axis='x', rotation=45)
        
        #Genes per cell distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.adata.obs['n_genes_by_counts'], bins=50, color='coral')
        ax2.set_title('Genes per Cell Distribution')
        ax2.set_xlabel('Number of Genes')
        ax2.set_ylabel('Frequency')
        
        #UMI counts distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.adata.obs['total_counts'], bins=50, color='mediumseagreen')
        ax3.set_title('UMI Counts Distribution')
        ax3.set_xlabel('Total Counts')
        ax3.set_ylabel('Frequency')
        
        #MT% distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(self.adata.obs['pct_counts_mt'], bins=50, color='orchid')
        ax4.set_title('Mitochondrial % Distribution')
        ax4.set_xlabel('% MT Counts')
        ax4.set_ylabel('Frequency')
        
        #Cluster sizes
        ax5 = fig.add_subplot(gs[1, 1])
        if 'leiden' in self.adata.obs.columns:
            cluster_sizes = self.adata.obs['leiden'].value_counts().sort_index()
            cluster_sizes.plot(kind='bar', ax=ax5, color='teal')
            ax5.set_title('Cluster Sizes')
            ax5.set_xlabel('Cluster')
            ax5.set_ylabel('Number of Cells')
        
        #Sample composition by cluster
        ax6 = fig.add_subplot(gs[1, 2])
        if 'leiden' in self.adata.obs.columns:
            comp = pd.crosstab(self.adata.obs['leiden'], 
                              self.adata.obs['sample'], 
                              normalize='index')
            comp.plot(kind='bar', stacked=True, ax=ax6, legend=True)
            ax6.set_title('Cluster Composition by Sample')
            ax6.set_xlabel('Cluster')
            ax6.set_ylabel('Proportion')
            ax6.legend(title='Sample', bbox_to_anchor=(1.05, 1))
        
        #UMAP plots
        if 'X_umap' in self.adata.obsm:
            #by cluster
            ax7 = fig.add_subplot(gs[2, 0])
            if 'leiden' in self.adata.obs.columns:
                for cluster in self.adata.obs['leiden'].unique():
                    mask = self.adata.obs['leiden'] == cluster
                    ax7.scatter(self.adata.obsm['X_umap'][mask, 0],
                               self.adata.obsm['X_umap'][mask, 1],
                               s=1, alpha=0.5, label=cluster)
                ax7.set_title('UMAP by Cluster')
                ax7.legend(markerscale=5, fontsize=8)
            
            #by sample
            ax8 = fig.add_subplot(gs[2, 1])
            for sample in self.adata.obs['sample'].unique():
                mask = self.adata.obs['sample'] == sample
                ax8.scatter(self.adata.obsm['X_umap'][mask, 0],
                           self.adata.obsm['X_umap'][mask, 1],
                           s=1, alpha=0.5, label=sample)
            ax8.set_title('UMAP by Sample')
            ax8.legend(markerscale=5, fontsize=8)
            
            #by n_genes
            ax9 = fig.add_subplot(gs[2, 2])
            sc = ax9.scatter(self.adata.obsm['X_umap'][:, 0],
                            self.adata.obsm['X_umap'][:, 1],
                            c=self.adata.obs['n_genes_by_counts'],
                            s=1, alpha=0.5, cmap='viridis')
            ax9.set_title('UMAP by Gene Count')
            plt.colorbar(sc, ax=ax9, label='n_genes')
        
        plt.suptitle('Hepatoblastoma scRNA-seq QC Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{self.output_dir}/figures/qc_summary_comprehensive.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
      
#MAIN EXECUTION FOR ADVANCED ANALYSES

def run_advanced_analyses(adata, output_dir='hepatoblastoma_results'):

    #Compositional Analysis
    if 'cell_type' in adata.obs.columns:
        comp = ComposionalAnalysis(adata, output_dir)
        cell_counts = comp.prepare_data(cell_type_col='cell_type', condition_col='sample')
        
        if cell_counts is not None and SCCODA_AVAILABLE:
            try:
                comp.run_scoda_model(cell_counts, reference_cell_type='Hepatocytes')
            except Exception as e:
                print(f"scCODA model failed: {e}")
    
    #Trajectory Analysis
    traj = TrajectoryAnalysis(adata, output_dir)
    traj.run_paga(groupby='leiden', threshold=0.03)
    
    if 'cell_type' in adata.obs.columns:
        traj.plot_pseudotime(root_cell_type='Hepatic_Progenitors')
    
    #Additional Visualizations
    viz = AdditionalViz(adata, output_dir)
    viz.create_qc_summary_plot()
    viz.plot_cluster_proportions(group_by='sample', color_by='leiden')
    
    if 'cell_type' in adata.obs.columns:
        viz.plot_cluster_proportions(group_by='sample', color_by='cell_type')
    
    return adata

if __name__ == "__main__":
    #Load processed data
    print("Loading processed data...")
    adata = sc.read_h5ad('hepatoblastoma_results/hepatoblastoma_processed.h5ad')
    
    #Run advanced analyses
    adata = run_advanced_analyses(adata, output_dir='hepatoblastoma_results')
