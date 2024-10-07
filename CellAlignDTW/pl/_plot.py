import matplotlib.pyplot as plt

def plot_sigmoid_fits(aligned_obj):
    samples = aligned_obj.df[aligned_obj.sample_col].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(samples), figsize=(15, 5), sharey=True)
    for i, sample in enumerate(samples):
        ax = axes[i]
        sample_data = aligned_obj.df[aligned_obj.df[aligned_obj.sample_col] == sample]
        x_data = sample_data[aligned_obj.score_col].values
        y_data = sample_data['numeric_label'].values
        cutoff_point = aligned_obj.cutoff_points[sample]
        ax.scatter(x_data, y_data, label='Data', alpha=0.2)
        ax.axvline(x=cutoff_point, color='green', linestyle='--', label=f'Cutoff at x={cutoff_point:.2f}')
        ax.set_title(f'Sample: {sample}')
        ax.set_xlabel('Score')
        if i == 0:
            ax.set_ylabel('Label')
        ax.legend()
    plt.show()

def plot_summary_curve(summary_df, aggregated_curve, scores, gene):
    plt.figure(figsize=(5, 4))
    summary_df = summary_df.loc[summary_df.gene == gene, :]
    scores = scores.loc[scores.gene == gene, :]
    
    # Plot each sample's smoothed curve
    for sample in summary_df['sample'].unique():
        sample_data = summary_df[summary_df['sample'] == sample]
        plt.plot(sample_data['aligned_score'], sample_data['smoothed'], alpha=0.3, label=f'{sample}')
    
    # Plot the aggregated curve
    plt.plot(aggregated_curve['aligned_score'], aggregated_curve[gene], color='black', linewidth=2, label='Aggregated Curve')
    
    plt.xlabel('Aligned score')
    plt.ylabel(f'{gene} z-score')
    plt.title(f'{gene} (MI = {round(scores['MI'].iloc[0], 3)}, MASE = {round(scores['MASE'].iloc[0], 3)}, Max @ {round(scores['Max'].iloc[0], 2)} %)')
    plt.legend()
    plt.show()

