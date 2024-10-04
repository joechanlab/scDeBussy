import scanpy as sc
import pandas as pd

def create_cellrank_probability_df(adata_paths, 
                                   cell_type_col,
                                   sample_names, 
                                   cellrank_cols_dict, 
                                   
                                   cellrank_obsm='term_states_fwd_memberships'):
    df_list = _extract_cellrank_probability(adata_paths, 
                                           cell_type_col,
                                           sample_names, 
                                           cellrank_cols_dict, 
                                           cellrank_obsm)
    df = pd.concat(df_list.values(), axis=0)
    return df

def _extract_cellrank_probability(adata_paths, 
                                 cell_type_col,
                                 sample_names, 
                                 cellrank_cols_dict, 
                                 cellrank_obsm):
    df_list = {}
    for i, adata_path in enumerate(adata_paths):
        sample_name = sample_names[i]
        adata = sc.read_h5ad(adata_path)
        df = pd.DataFrame(adata.obsm[cellrank_obsm],
                          columns=cellrank_cols_dict[sample_name],
                          index=adata.obs_names)
        df['sample'] = sample_name
        df['cell_id'] = adata.obs_names
        df['cell_type'] = adata.obs[cell_type_col]
        df_long = pd.melt(df, 
                        id_vars=['sample', 'cell_id', 'cell_type'], 
                        var_name='macrostate',
                        value_name='probability')
        df_long['state'] = df_long['macrostate'].apply(lambda x: x.split("_")[0])
        df_long = df_long.drop('macrostate', axis=1)
        df_long = df_long.groupby(['sample', 'cell_id', 'cell_type', 'state'], observed=True).max().dropna().reset_index()
        df = df_long.pivot(index=['sample', 'cell_id', 'cell_type'], columns='state', values='probability').reset_index()
        df_list[sample_name] = df
    return df_list

