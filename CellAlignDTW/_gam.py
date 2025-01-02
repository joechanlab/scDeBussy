import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import gc

def _remove_outliers(df, column, method='zscore', threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def gam_smooth_expression(df, genes, n_splines = 6, lam = 3):
    gene_curve = {}
    scores_list = []
    summary_df_list = []
    for i, gene in enumerate(tqdm(genes, desc="Processing genes")):
        summary_df, aggregated_curve, scores = _gam_smooth_expression(df, gene, n_splines, lam)
        if i == 0:
            gene_curve['aligned_score'] = aggregated_curve['aligned_score']
        gene_curve[gene] = aggregated_curve.smoothed
        summary_df['gene'] = gene
        summary_df_list.append(summary_df)
        scores_list.append(scores)
        del summary_df, aggregated_curve
        gc.collect()
    summary_df = pd.concat(summary_df_list, ignore_index=True)
    scores_df = pd.DataFrame(scores_list)
    scores_df = scores_df.sort_values(by=['Deviance', 'GCV', 'AIC', 'MI', 'Max'], ascending=[True, True, True, False, True])
    gene_curve = pd.DataFrame(gene_curve)
    return summary_df, gene_curve, scores_df

def _gam_smooth_expression(df, gene, subject_col='subject', score_col='aligned_score', n_splines=6, lam=3):
    # Remove outliers
    df = df[[subject_col, score_col, gene]]
    df.columns = [subject_col, score_col, 'expression']
    df = _remove_outliers(df.copy(), score_col, method='zscore')
    scaler = StandardScaler()
    df['expression'] = df.groupby('subject')['expression'].transform(lambda x: StandardScaler().fit_transform(x.values.reshape(-1, 1)).flatten())
    df.sort_values(by=score_col, inplace=True)

    # Extract features and target
    x = df[[score_col, 'subject']].copy()
    x['subject'] = x['subject'].astype('category').cat.codes
    y = df['expression'].values
    
    # Define and fit the GAM with subject as a factor
    gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) + f(1)).fit(x, y)
    # Add smoothed values to the DataFrame
    df['smoothed'] = gam.predict(x)
    df['deviance_residue'] = gam.deviance_residuals(x, y)
    subject_deviance_agg = df.groupby('subject')['deviance_residue'].mean().reset_index()

    # fit another GAM across the subjects
    # using the same sets of the x axis to produce the aggregated curve
    x_grid = np.linspace(df[score_col].min(), df[score_col].max(), 1000)
    prediction_dfs = []
    for subject in x['subject'].unique():
        grid_df = pd.DataFrame({score_col: x_grid})
        grid_df['subject'] = subject
        grid_df['smoothed'] = gam.predict(grid_df)
        prediction_dfs.append(grid_df)
    predictions_combined = pd.concat(prediction_dfs)
    aggregated_curve = predictions_combined.groupby(score_col)['smoothed'].mean().reset_index()

    # Compute evaluation metrics
    mi_score = compute_mutual_information(aggregated_curve, score_col, 'smoothed')
    max_point = maximum_point(aggregated_curve['smoothed'])
    total_deviance_mean = subject_deviance_agg['deviance_residue'].mean()
    
    scores = {
        'gene': gene,
        "GCV": gam.statistics_['GCV'],
        "AIC": gam.statistics_['AIC'],
        "MI": mi_score,
        "Max": max_point * 100,
        "Deviance": total_deviance_mean
    }
    scores.update()
    return df, aggregated_curve, scores

def compute_mutual_information(df, aligned_col, aggregated_col, bins=10):
    aligned_discrete = pd.cut(df[aligned_col], bins=bins, labels=False)
    aggregated_discrete = pd.cut(df[aggregated_col], bins=bins, labels=False)
    mi = mutual_info_score(aligned_discrete, aggregated_discrete)
    return mi

def maximum_point(expression_data):
    return np.argmax(expression_data)/len(expression_data)
