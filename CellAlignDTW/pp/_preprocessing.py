import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy

def calculate_composite_score(df, score_columns):
    # normalize each row to 1
    df[score_columns] = df[score_columns].div(df[score_columns].sum(axis=1), axis=0)
    # Assign weights to each cell type based on their order
    num_scores = len(score_columns)
    weights = [i / (num_scores - 1) for i in range(num_scores)]
    
    # Weighted sum of the probabilities, reflecting the transition
    df['score'] = sum(df[col] * weight for col, weight in zip(score_columns, weights))
    return df

def remove_outliers(df, columns, threshold=3):
    if not isinstance(columns, list):
        columns = [columns]
    n = df.shape[0]
    for column in columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]
    if n - df.shape[0] > 0:
        print(f'Removed {n - df.shape[0]} outliers from column {columns}...')
    return df

def stratified_downsample(df, score_cols, downsample_size, seed=42):
    if not isinstance(score_cols, list):
        score_cols = [score_cols]
    df['score_bins'] = pd.cut(df[score_cols].max(axis=1), bins=len(score_cols))
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

def denoise_cell_type(df, cluster_ordering, original_assignments, threshold=0.1):
    # Iterate over each row to assign cell types
    for index, row in df.iterrows():
        # Sort probabilities in descending order
        sorted_probs = row[cluster_ordering].sort_values(ascending=False)
        # Check if the top two probabilities are close
        if (sorted_probs.iloc[0] - sorted_probs.iloc[1]) < threshold:
            # Trust original assignment if values are close
            df.at[index, 'cell_type'] = original_assignments[index]
        else:
            # Assign based on maximum probability
            df.at[index, 'cell_type'] = sorted_probs.idxmax()
    return df

def create_cellrank_probability_df(adata_paths, 
                                   cell_type_col,
                                   subject_names,
                                   cluster_ordering,
                                   pseudotime_col,
                                   cellrank_cols_dict=None, 
                                   downsample=np.Inf,
                                   layer=None,
                                   seed=42):
    df_dict = {}
    adata_list = []
    subject_ordering = []
    for i, adata_path in enumerate(adata_paths):
        subject_name = subject_names[i]
        adata = sc.read_h5ad(adata_path)
        if 'SCLC-AN' in cluster_ordering:
            adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).replace('SCLC-[AN]', 'SCLC-AN', regex=True)
        adata.obs['subject'] = subject_name
        if (not all(np.isin(cluster_ordering, adata.obs[cell_type_col].unique()))) or ((not pseudotime_col in adata.obsm.keys()) and (not pseudotime_col in adata.obs.keys())):
            print(f"Skipping {subject_name} due to missing cell types")
            continue
        else: 
            subject_ordering.append(subject_name)
            print(f"Processing {subject_name}")
            adata = adata[np.isin(adata.obs[cell_type_col], cluster_ordering),:].copy()
            if pseudotime_col == 'term_states_fwd_memberships':
                df = pd.DataFrame(adata.obsm[pseudotime_col],
                                columns=cellrank_cols_dict[subject_name],
                                index=adata.obs_names)
                df['subject'] = subject_name
                df['cell_id'] = adata.obs_names
                df['cell_type'] = adata.obs[cell_type_col].astype(str)
                df_long = pd.melt(df, 
                                id_vars=['subject', 'cell_id', 'cell_type'], 
                                var_name='macrostate',
                                value_name='probability')
                df_long['state'] = df_long['macrostate'].apply(lambda x: x.split("_")[0])
                if 'SCLC-AN' in cluster_ordering:
                    df_long['state'] = df_long['state'].replace('SCLC-[AN]', 'SCLC-AN', regex=True)
                df_long = df_long.drop('macrostate', axis=1)
                df_long = df_long.groupby(['subject', 'cell_id', 'cell_type', 'state'], observed=True).max().dropna().reset_index()
                df = df_long.pivot(index=['subject', 'cell_id', 'cell_type'], columns='state', values='probability').reset_index()
                df['cell_type'] = df[cluster_ordering].idxmax(axis=1)
                df = df.loc[np.isin(df['cell_type'], cluster_ordering),:]
                adata = adata[np.isin(adata.obs_names, df.cell_id),:].copy()
                df_cell_type = pd.DataFrame({'cell_type': df.cell_type.values}, index=df.cell_id)
                df_cell_type = df_cell_type.loc[adata.obs_names]
                adata.obs['cell_type_corrected'] = df_cell_type.cell_type.values
                df = calculate_composite_score(df, cluster_ordering)

            elif 'palantir_pseudotime' in pseudotime_col:
                df = pd.DataFrame(adata.obs[pseudotime_col].values,
                    columns=['score'],
                    index=adata.obs_names)
                df['subject'] = subject_name
                df['cell_id'] = adata.obs_names
                df['cell_type'] = adata.obs[cell_type_col].astype(str)

            label_mapping = {label: idx for idx, label in enumerate(cluster_ordering)}
            df['numeric_label'] = df['cell_type'].map(label_mapping)
            #df = remove_outliers(df, 'score', threshold=3)
            n = df.shape[0]
            if df.shape[0] > downsample:
                np.random.seed(seed)
                df = stratified_downsample(df, 'score', downsample_size=downsample, seed=seed)
                print(f'Downsampled {subject_name} from {n} to {df.shape[0]} cells')
            adata_list.append(adata[df['cell_id'],:])
            df_dict[subject_name] = df
    score_df = pd.concat(df_dict, axis=0)
    score_df = score_df.reset_index(drop=True)
    combined_adata = ad.concat(adata_list, join='outer', keys=subject_names)
    if layer is not None:
        X = combined_adata.layers[layer] #'MAGIC_imputed_data'
    else:
        X = combined_adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    gene_df = pd.DataFrame(X,
                          index = combined_adata.obs_names,
                          columns = combined_adata.var_names)
    gene_df = gene_df.reset_index(names='cell_id')
    df = score_df.merge(gene_df, on = ['cell_id'])
    df['subject'] = pd.Categorical(df['subject'], categories=subject_ordering, ordered=True)
    df = df.sort_values(['subject', 'score'])
    return combined_adata, df
