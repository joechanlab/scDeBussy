import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

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
        if 'SCLC-A/N' in cluster_ordering:
            adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).replace('SCLC-[AN]', 'SCLC-A/N', regex=True)
        adata.obs['sample'] = sample_name
        if (not all(np.isin(cluster_ordering, adata.obs[cell_type_col].unique()))) or (not cellrank_obsm in adata.obsm.keys()):
            print(f"Skipping {sample_name} due to missing cell types")
            continue
        else: 
            print(f"Processing {sample_name}")
            adata = adata[np.isin(adata.obs[cell_type_col], cluster_ordering),:]
            if adata.shape[0] > downsample:
                print(f'Downsampling {sample_name} to {downsample} cells')
                np.random.seed(seed)
                adata = adata[np.random.choice(adata.obs_names, downsample, replace=False),:]
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
            if 'SCLC-A/N' in cluster_ordering:
                df_long['state'] = df_long['state'].replace('SCLC-[AN]', 'SCLC-A/N', regex=True)
            df_long = df_long.drop('macrostate', axis=1)
            df_long = df_long.groupby(['sample', 'cell_id', 'cell_type', 'state'], observed=True).max().dropna().reset_index()
            df = df_long.pivot(index=['sample', 'cell_id', 'cell_type'], columns='state', values='probability').reset_index()
            adata_list.append(adata[df['cell_id'],:])
            df_dict[sample_name] = df
    score_df = pd.concat(df_dict, axis=0)
    score_df.loc[:, 'score'] = score_df.loc[:,cluster_ordering[1]] - score_df.loc[:,cluster_ordering[0]]
    score_df = score_df.reset_index(drop=True)
    combined_adata = ad.concat(adata_list, join='outer', keys=sample_names)
    gene_df = combined_adata.to_df()
    gene_df = gene_df.reset_index(names='cell_id')
    df = score_df.merge(gene_df, on = ['cell_id'])
    return df
