import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedShuffleSplit

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    print(f'Removing {np.sum(z_scores >= threshold)} outliers...')
    return df[z_scores < threshold]

def stratified_downsample(df, score_col, downsample_size, seed=42):
    df['score_bins'] = pd.cut(df[score_col], bins=100)
    bin_counts = df['score_bins'].value_counts()
    initial_samples_per_bin = downsample_size // len(bin_counts)
    df_sampled = pd.DataFrame()
    np.random.seed(seed)
    
    # Sample equally from each bin initially
    for bin_label in bin_counts.index:
        bin_data = df[df['score_bins'] == bin_label]
        sampled_data = bin_data.sample(n=min(len(bin_data), initial_samples_per_bin), random_state=seed)
        df_sampled = pd.concat([df_sampled, sampled_data], axis=0)
    
    # Drop the temporary 'score_bins' column
    df_sampled.drop(columns='score_bins', inplace=True)
    
    return df_sampled

def create_cellrank_probability_df(adata_paths, 
                                   cell_type_col,
                                   sample_names, 
                                   cellrank_cols_dict, 
                                   cluster_ordering,
                                   cellrank_obsm='term_states_fwd_memberships',
                                   downsample=5000,
                                   seed=42):
    df_dict = {}
    adata_list = []
    for i, adata_path in enumerate(adata_paths):
        sample_name = sample_names[i]
        adata = sc.read_h5ad(adata_path)
        if 'SCLC-AN' in cluster_ordering:
            adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).replace('SCLC-[AN]', 'SCLC-AN', regex=True)
        adata.obs['sample'] = sample_name
        if (not all(np.isin(cluster_ordering, adata.obs[cell_type_col].unique()))) or (not cellrank_obsm in adata.obsm.keys()):
            print(f"Skipping {sample_name} due to missing cell types")
            continue
        else: 
            print(f"Processing {sample_name}")
            adata = adata[np.isin(adata.obs[cell_type_col], cluster_ordering),:]
            df = pd.DataFrame(adata.obsm[cellrank_obsm],
                            columns=cellrank_cols_dict[sample_name],
                            index=adata.obs_names)
            df['sample'] = sample_name
            df['cell_id'] = adata.obs_names
            df['cell_type'] = adata.obs[cell_type_col].astype(str)
            df_long = pd.melt(df, 
                            id_vars=['sample', 'cell_id', 'cell_type'], 
                            var_name='macrostate',
                            value_name='probability')
            df_long['state'] = df_long['macrostate'].apply(lambda x: x.split("_")[0])
            if 'SCLC-AN' in cluster_ordering:
                df_long['state'] = df_long['state'].replace('SCLC-[AN]', 'SCLC-AN', regex=True)
            df_long = df_long.drop('macrostate', axis=1)
            df_long = df_long.groupby(['sample', 'cell_id', 'cell_type', 'state'], observed=True).max().dropna().reset_index()
            df = df_long.pivot(index=['sample', 'cell_id', 'cell_type'], columns='state', values='probability').reset_index()
            df.loc[:, 'score'] = df.loc[:,cluster_ordering[1]] - df.loc[:,cluster_ordering[0]]
            df = remove_outliers(df, 'score')
            
            if df.shape[0] > downsample:
                np.random.seed(seed)
                df = stratified_downsample(df, 'score', downsample_size=downsample, seed=seed)
                print(f'Downsampled {sample_name} to {df.shape[0]} cells')
                
            adata_list.append(adata[df['cell_id'],:])
            df_dict[sample_name] = df
    score_df = pd.concat(df_dict, axis=0)
    score_df = score_df.reset_index(drop=True)
    combined_adata = ad.concat(adata_list, join='outer', keys=sample_names)
    gene_df = pd.DataFrame(combined_adata.layers['MAGIC_imputed_data'],
                          index = combined_adata.obs_names,
                          columns = combined_adata.var_names)
    gene_df = gene_df.reset_index(names='cell_id')
    df = score_df.merge(gene_df, on = ['cell_id'])
    return df
